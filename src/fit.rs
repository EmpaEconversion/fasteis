use std::cell::RefCell;
use std::f64::consts::TAU;

use argmin::core::{CostFunction, Executor, State};
use argmin::solver::neldermead::NelderMead;
use argmin::solver::particleswarm::ParticleSwarm;
use argmin::solver::simulatedannealing::{Anneal, SimulatedAnnealing};
use differential_evolution::self_adaptive_de;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::circuit::{self, Node, Series};

#[derive(Clone, Copy, Debug)]
pub enum Weighting {
    Unit,
    Modulus,
}

#[derive(Clone, Copy, Debug)]
pub struct FitOptions {
    pub max_iterations: u32,
    pub ftol: f64,
    pub xtol: f64,
    pub gtol: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions { max_iterations: 200, ftol: 1e-8, xtol: 1e-8, gtol: 1e-8 }
    }
}

#[derive(Debug)]
pub struct FitOutcome {
    pub node: Series,
    pub param_names: Vec<String>,
    pub params: Vec<f64>,
    pub success: bool,
    pub iterations: u64,
    pub cost: f64,
    pub chi_square: f64,
    pub stderr: Option<Vec<f64>>,
}

#[derive(Debug)]
pub enum FitError {
    LengthMismatch,
    EmptyData,
    NoFreeParameters,
    SolverError(String),
    UnknownMethod(String),
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::LengthMismatch => f.write_str("frequencies and impedances must have the same length"),
            FitError::EmptyData => f.write_str("frequencies and impedances must not be empty"),
            FitError::NoFreeParameters => f.write_str("circuit topology has no parameters to fit"),
            FitError::SolverError(msg) => write!(f, "solver error: {msg}"),
            FitError::UnknownMethod(m) => {
                write!(
                    f,
                    "unknown fit method {m:?}, expected one of \"levenberg_marquardt\", \"particle_swarm\", \
                     \"nelder_mead\", \"differential_evolution\", \"simulated_annealing\", \"basin_hopping\""
                )
            }
        }
    }
}

pub(crate) fn compute_weights(z_measured: &[Complex64], weighting: Weighting) -> Vec<f64> {
    match weighting {
        Weighting::Unit => vec![1.0; z_measured.len()],
        Weighting::Modulus => z_measured.iter().map(|z| z.norm().max(1e-30)).collect(),
    }
}

/// Interleaved `[re0, im0, re1, im1, ...]` weighted residual vector, length `2N`.
/// Optimizer-agnostic: operates on plain `Vec<f64>`, reused unchanged by any future
/// non-LM backend (see the `argmin` extensibility note in the fit() design).
pub(crate) fn residuals(
    topology: &[Node],
    p: &[f64],
    omegas: &[f64],
    z_measured: &[Complex64],
    weights: &[f64],
) -> Vec<f64> {
    let mut r = vec![0.0; 2 * omegas.len()];
    for (i, &omega) in omegas.iter().enumerate() {
        let z = circuit::impedance_with_params(topology, p, omega);
        let w = weights[i];
        r[2 * i] = (z.re - z_measured[i].re) / w;
        r[2 * i + 1] = (z.im - z_measured[i].im) / w;
    }
    r
}

/// Central-difference Jacobian columns.
/// Perturbations are not clamped to physical bounds: impedance() is smooth well outside
/// those ranges, so clamping here would bias the derivative estimate near a boundary.
pub(crate) fn jacobian_columns(
    topology: &[Node],
    p: &[f64],
    omegas: &[f64],
    z_measured: &[Complex64],
    weights: &[f64],
) -> Vec<Vec<f64>> {
    (0..p.len())
        .into_par_iter()
        .map(|j| {
            let h = f64::EPSILON.cbrt() * p[j].abs().max(1e-8);
            let mut p_plus = p.to_vec();
            p_plus[j] += h;
            let mut p_minus = p.to_vec();
            p_minus[j] -= h;
            let r_plus = residuals(topology, &p_plus, omegas, z_measured, weights);
            let r_minus = residuals(topology, &p_minus, omegas, z_measured, weights);
            r_plus.iter().zip(&r_minus).map(|(a, b)| (a - b) / (2.0 * h)).collect()
        })
        .collect()
}

/// Adapter implementing `LeastSquaresProblem` for the `levenberg-marquardt` crate.
/// Operates in log-coordinate space for open-bound (lower-bound-only) parameters --
/// reusing the same `to_pso_coord`/`from_pso_coord` transform PSO uses -- and raw
/// physical space for double-bounded ones (alpha/gamma).
struct LmProblem<'a> {
    topology: &'a [Node],
    omegas: Vec<f64>,
    z_measured: &'a [Complex64],
    weights: Vec<f64>,
    bounds: Vec<(f64, f64)>,
    coord: DVector<f64>,
    /// Parameters held at their starting coordinate for the whole run, via a
    /// zeroed Jacobian column in `jacobian()`.
    fixed: Vec<bool>,
}

impl LmProblem<'_> {
    fn physical(&self) -> Vec<f64> {
        self.coord.iter().zip(&self.bounds).map(|(&c, &b)| from_pso_coord(c, b)).collect()
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for LmProblem<'_> {
    type ParameterStorage = nalgebra::storage::Owned<f64, Dyn>;
    type ResidualStorage = nalgebra::storage::Owned<f64, Dyn>;
    type JacobianStorage = nalgebra::storage::Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, c: &DVector<f64>) {
        self.coord = c.clone();
    }

    fn params(&self) -> DVector<f64> {
        self.coord.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let p = self.physical();
        Some(DVector::from_vec(residuals(self.topology, &p, &self.omegas, self.z_measured, &self.weights)))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let p = self.physical();
        let cols = jacobian_columns(self.topology, &p, &self.omegas, self.z_measured, &self.weights);
        let m = cols.first()?.len();
        let n = cols.len();
        Some(DMatrix::from_fn(m, n, |i, j| {
            if self.fixed[j] {
                return 0.0;
            }
            let (lo, hi) = self.bounds[j];
            // d(physical)/d(coord): identity for double-bounded (hi finite), else
            // p = lo + 10^c => dp/dc = ln(10) * 10^c = ln(10) * (p - lo).
            let dp_dc = if hi.is_finite() { 1.0 } else { (p[j] - lo) * std::f64::consts::LN_10 };
            cols[j][i] * dp_dc
        }))
    }
}

/// One LM run from a specific starting coordinate vector; returns (params, success,
/// evaluations). `fixed[j] == true` holds parameter `j` at `start_coord[j]` for the
/// whole run; pass an all-`false` mask for a normal, fully free run.
fn levenberg_marquardt_single_start(
    topology: &[Node],
    omegas: &[f64],
    z_measured: &[Complex64],
    weights: &[f64],
    bounds: &[(f64, f64)],
    start_coord: DVector<f64>,
    fixed: &[bool],
    options: &FitOptions,
) -> (Vec<f64>, bool, u64) {
    let problem = LmProblem {
        topology,
        omegas: omegas.to_vec(),
        z_measured,
        weights: weights.to_vec(),
        bounds: bounds.to_vec(),
        coord: start_coord,
        fixed: fixed.to_vec(),
    };

    let solver = LevenbergMarquardt::new()
        .with_ftol(options.ftol)
        .with_xtol(options.xtol)
        .with_gtol(options.gtol)
        .with_patience(options.max_iterations.max(1) as usize);

    let (solved, report) = solver.minimize(problem);
    let success = report.termination.was_successful();
    let params: Vec<f64> = solved.params().iter().zip(bounds).map(|(&c, &b)| from_pso_coord(c, b)).collect();
    (params, success, report.number_of_evaluations as u64)
}

/// Multiplicative perturbations applied to open-bound (lower-bound-only)
/// parameters when generating candidate starting points, alongside the caller's
/// own guess (factor 1.0).
const LM_RESTART_FACTORS: [f64; 5] = [1.0, 3.0, 1.0 / 3.0, 8.0, 1.0 / 8.0];

/// Spread tried for double-bounded parameters (alpha/gamma) when generating
/// candidate starting points, alongside the caller's own guess.
const ALPHA_GAMMA_PROBES: [f64; 4] = [0.2, 0.4, 0.6, 0.8];

/// A candidate starting point: physical-space parameter values, plus its cost at a
/// single `residuals()` evaluation (no optimization).
struct Candidate {
    params: Vec<f64>,
    screening_cost: f64,
}

/// Candidate starting points for `levenberg_marquardt_fit`, ranked by screening
/// cost. Sweeps the open-bound parameters (`LM_RESTART_FACTORS`) and the
/// double-bounded parameters (`ALPHA_GAMMA_PROBES`) independently, each holding
/// the other group at the caller's guess.
fn candidate_starting_points(
    topology: &[Node],
    p0: &[f64],
    bounds: &[(f64, f64)],
    omegas: &[f64],
    z_measured: &[Complex64],
    weights: &[f64],
) -> Vec<Candidate> {
    let screen = |params: Vec<f64>| {
        let r = residuals(topology, &params, omegas, z_measured, weights);
        let cost = 0.5 * r.iter().map(|x| x * x).sum::<f64>();
        Candidate { params, screening_cost: cost }
    };

    let mut candidates = vec![screen(p0.to_vec())];

    for &factor in &LM_RESTART_FACTORS[1..] {
        let params: Vec<f64> =
            p0.iter().zip(bounds).map(|(&p, &(_lo, hi))| if hi.is_finite() { p } else { p * factor }).collect();
        candidates.push(screen(params));
    }

    for &probe in &ALPHA_GAMMA_PROBES {
        let params: Vec<f64> = p0
            .iter()
            .zip(bounds)
            .map(|(&p, &(lo, hi))| if hi.is_finite() { lo + probe * (hi - lo) } else { p })
            .collect();
        candidates.push(screen(params));
    }

    candidates.sort_by(|a, b| a.screening_cost.total_cmp(&b.screening_cost));
    candidates
}

/// Distance from either edge of `[0, 1]`, where a double-bounded parameter
/// counts as pinned in `looks_reliable`.
const BOUND_PIN_TOLERANCE: f64 = 1e-6;

/// Multiplicative distance from its starting value beyond which an open-bound
/// parameter counts as runaway in `looks_reliable`.
const RUNAWAY_RATIO: f64 = 1e6;

/// Multiplicative distance from its lower bound within which an open-bound
/// parameter counts as pinned in `looks_reliable`. Note that `lo` is a fixed
/// numerical floor (`1e-12`, see `Element::param_bounds`).
const LOWER_BOUND_PIN_FACTOR: f64 = 100.0;

/// Judge whether a converged value is trustworthy enough to stop restarting on.
/// If the solver says not converged, or any parameter landed pinned or runaway,
/// then more restarts are attempted.
fn looks_reliable(params: &[f64], p0: &[f64], bounds: &[(f64, f64)], success: bool, cost: f64) -> bool {
    if !success || !cost.is_finite() {
        return false;
    }
    params.iter().zip(p0).zip(bounds).all(|((&p, &p0), &(lo, hi))| {
        if hi.is_finite() {
            let span = hi - lo;
            (p - lo).abs() > BOUND_PIN_TOLERANCE * span && (hi - p).abs() > BOUND_PIN_TOLERANCE * span
        } else {
            let ratio = p / p0.abs().max(1e-300);
            p > lo * LOWER_BOUND_PIN_FACTOR
                && ratio.is_finite()
                && (1.0 / RUNAWAY_RATIO..RUNAWAY_RATIO).contains(&ratio)
        }
    })
}

/// Fit a circuit to impedance data with Levenberg-Marquardt.
/// Starts with several candidate starting points.
/// Fits most promising first.
/// Stops when fit converged and `looks_reliable` trusts the result.
/// Fallback to whichever run had the lowest cost.
/// If result has a double-bounded paramter pinned at/beyong its bound, repolish
/// with the parameter clamped and the rest refit, keep only if fit improves.
pub fn levenberg_marquardt_fit(
    topology: &[Node],
    frequencies: &[f64],
    z_measured: &[Complex64],
    weighting: Weighting,
    options: &FitOptions,
) -> Result<FitOutcome, FitError> {
    if frequencies.len() != z_measured.len() {
        return Err(FitError::LengthMismatch);
    }
    if frequencies.is_empty() {
        return Err(FitError::EmptyData);
    }
    if circuit::param_count(topology) == 0 {
        return Err(FitError::NoFreeParameters);
    }

    let omegas: Vec<f64> = frequencies.iter().map(|f| TAU * f).collect();
    let weights = compute_weights(z_measured, weighting);
    let bounds = circuit::param_bounds(topology);
    let p0 = circuit::param_values(topology);

    let candidates = candidate_starting_points(topology, &p0, &bounds, &omegas, z_measured, &weights);

    let mut best: Option<(Vec<f64>, bool, f64)> = None;
    let mut total_evaluations = 0u64;

    let no_fixed = vec![false; p0.len()];
    for candidate in &candidates {
        let start_coord: Vec<f64> =
            candidate.params.iter().zip(&bounds).map(|(&p, &b)| to_pso_coord(p, b)).collect();

        let (params, success, evaluations) = levenberg_marquardt_single_start(
            topology,
            &omegas,
            z_measured,
            &weights,
            &bounds,
            DVector::from_vec(start_coord),
            &no_fixed,
            options,
        );
        total_evaluations += evaluations;

        let cost = {
            let r = residuals(topology, &params, &omegas, z_measured, &weights);
            0.5 * r.iter().map(|x| x * x).sum::<f64>()
        };

        let is_better = best.as_ref().is_none_or(|(_, best_success, best_cost)| {
            (success && !*best_success) || (success == *best_success && cost < *best_cost)
        });
        let reliable = looks_reliable(&params, &p0, &bounds, success, cost);
        if is_better {
            best = Some((params, success, cost));
        }
        if reliable {
            break;
        }
    }

    let (params, success, cost) = best.expect("candidate_starting_points always returns at least one candidate");

    let clamped: Vec<f64> = params.iter().zip(&bounds).map(|(&p, &(lo, hi))| p.clamp(lo, hi)).collect();
    let fixed: Vec<bool> =
        params.iter().zip(&bounds).map(|(&p, &(lo, hi))| hi.is_finite() && (p <= lo || p >= hi)).collect();

    let (final_params, final_success) = if fixed.iter().any(|&f| f) {
        let start_coord: Vec<f64> = clamped.iter().zip(&bounds).map(|(&p, &b)| to_pso_coord(p, b)).collect();
        let (repolished, repolished_success, repolish_evals) = levenberg_marquardt_single_start(
            topology,
            &omegas,
            z_measured,
            &weights,
            &bounds,
            DVector::from_vec(start_coord),
            &fixed,
            options,
        );
        total_evaluations += repolish_evals;

        let repolished_cost = {
            let r = residuals(topology, &repolished, &omegas, z_measured, &weights);
            0.5 * r.iter().map(|x| x * x).sum::<f64>()
        };
        if repolished_success && repolished_cost < cost {
            (repolished, true)
        } else {
            (clamped, success)
        }
    } else {
        (params, success)
    };

    Ok(build_outcome(topology, final_params, final_success, total_evaluations, &omegas, z_measured, &weights))
}

/// Assemble a `FitOutcome` from a raw parameter vector: cost/chi_square/stderr
/// are always recomputed as the params may not come directly from that solver's
/// own endpoint.
fn build_outcome(
    topology: &[Node],
    params: Vec<f64>,
    success: bool,
    iterations: u64,
    omegas: &[f64],
    z_measured: &[Complex64],
    weights: &[f64],
) -> FitOutcome {
    // Final safety clamp into physical bounds as methods can produce a slightly
    // out-of-range value.
    let bounds = circuit::param_bounds(topology);
    let params: Vec<f64> = params.into_iter().zip(&bounds).map(|(v, &(lo, hi))| v.clamp(lo, hi)).collect();

    let cost = {
        let r = residuals(topology, &params, omegas, z_measured, weights);
        0.5 * r.iter().map(|x| x * x).sum::<f64>()
    };
    let chi_square = 2.0 * cost;

    let stderr = {
        let cols = jacobian_columns(topology, &params, omegas, z_measured, weights);
        let m = cols.first().map_or(0, Vec::len);
        let n = cols.len();
        let j = DMatrix::from_fn(m, n, |i, jc| cols[jc][i]);
        let dof = m as f64 - n as f64;
        if dof <= 0.0 {
            None
        } else {
            let jtj = j.transpose() * &j;
            jtj.try_inverse().map(|inv| {
                let scale = chi_square / dof;
                (0..inv.nrows()).map(|i| (inv[(i, i)] * scale).sqrt()).collect()
            })
        }
    };

    FitOutcome {
        node: circuit::with_param_values(topology, &params),
        param_names: circuit::param_names(topology),
        params,
        success,
        iterations,
        cost,
        chi_square,
        stderr,
    }
}

/// Coordinate PSO actually searches in for one parameter: `log10(p - lo)` for
/// an open-ended (lower-bound-only) parameter e.g. R/C/L/Q, `p` itself
/// (identity) for a double-bounded one (e.g. alpha/gamma).
fn to_pso_coord(p: f64, (lo, hi): (f64, f64)) -> f64 {
    if hi.is_finite() { p } else { (p - lo).max(1e-300).log10() }
}

fn from_pso_coord(c: f64, (lo, hi): (f64, f64)) -> f64 {
    if hi.is_finite() { c } else { lo + 10f64.powf(c) }
}

/// `argmin::core::CostFunction` adapter for particle swarm optimization.
/// `Param` here is the PSO-coordinate vector (see `to_pso_coord`), converted to
/// physical parameters via `bounds` before evaluating the model.
struct PsoProblem<'a> {
    topology: &'a [Node],
    omegas: &'a [f64],
    z_measured: &'a [Complex64],
    weights: &'a [f64],
    bounds: &'a [(f64, f64)],
}

impl CostFunction for PsoProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, coord: &Vec<f64>) -> Result<f64, argmin::core::Error> {
        let p: Vec<f64> = coord.iter().zip(self.bounds).map(|(&c, &b)| from_pso_coord(c, b)).collect();
        let r = residuals(self.topology, &p, self.omegas, self.z_measured, self.weights);
        Ok(0.5 * r.iter().map(|x| x * x).sum::<f64>())
    }
}

/// Derive a finite search box, in PSO-coordinate space, from each parameter's
/// physical bounds and its current value. PSO requires both endpoints, so for
/// singly-bounded parameters center a wide (8-decade) window on the current
/// value, then convert that window to PSO-coordinate space.
/// Double-bounded parameters just get their native `[lo, hi]` range directly.
fn pso_search_box(bounds: &[(f64, f64)], guess: &[f64]) -> (Vec<f64>, Vec<f64>) {
    bounds
        .iter()
        .zip(guess)
        .map(|(&b @ (lo, hi), &g)| {
            if hi.is_finite() {
                (lo, hi)
            } else {
                let g = g.abs().max(1e-6);
                let window = ((g * 1e-4).max(lo), g * 1e4);
                (to_pso_coord(window.0, b), to_pso_coord(window.1, b))
            }
        })
        .unzip()
}

/// Fit via particle swarm optimization followed by a LM polish.
/// Pass `Some(seed)` for a reproducible run.
pub fn particle_swarm_fit(
    topology: &[Node],
    frequencies: &[f64],
    z_measured: &[Complex64],
    weighting: Weighting,
    num_particles: usize,
    generations: u64,
    seed: Option<u64>,
) -> Result<FitOutcome, FitError> {
    if frequencies.len() != z_measured.len() {
        return Err(FitError::LengthMismatch);
    }
    if frequencies.is_empty() {
        return Err(FitError::EmptyData);
    }
    if circuit::param_count(topology) == 0 {
        return Err(FitError::NoFreeParameters);
    }

    let omegas: Vec<f64> = frequencies.iter().map(|f| TAU * f).collect();
    let weights = compute_weights(z_measured, weighting);
    let bounds = circuit::param_bounds(topology);
    let guess = circuit::param_values(topology);
    let (lower, upper) = pso_search_box(&bounds, &guess);

    let rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_os_rng(),
    };
    let problem = PsoProblem { topology, omegas: &omegas, z_measured, weights: &weights, bounds: &bounds };
    let solver = ParticleSwarm::new((lower, upper), num_particles).with_rng_generator(rng);

    let result = Executor::new(problem, solver)
        .configure(|state| state.max_iters(generations))
        .run()
        .map_err(|e| FitError::SolverError(e.to_string()))?;

    let best_coord = result
        .state
        .get_best_param()
        .map(|particle| particle.position.clone())
        .ok_or_else(|| FitError::SolverError("particle swarm found no best position".to_string()))?;
    let best: Vec<f64> = best_coord.iter().zip(&bounds).map(|(&c, &b)| from_pso_coord(c, b)).collect();
    let pso_evaluations = result.state.get_iter();

    polish_or_fallback(topology, best, pso_evaluations, frequencies, z_measured, weighting, &omegas, &weights)
}

/// Polish a candidate parameter vector with local unconstrained LM.
#[allow(clippy::too_many_arguments)]
fn polish_or_fallback(
    topology: &[Node],
    candidate: Vec<f64>,
    candidate_iterations: u64,
    frequencies: &[f64],
    z_measured: &[Complex64],
    weighting: Weighting,
    omegas: &[f64],
    weights: &[f64],
) -> Result<FitOutcome, FitError> {
    let candidate_outcome =
        build_outcome(topology, candidate.clone(), true, candidate_iterations, omegas, z_measured, weights);

    let polish_topology = circuit::with_param_values(topology, &candidate);
    let polished =
        levenberg_marquardt_fit(&polish_topology, frequencies, z_measured, weighting, &FitOptions::default())?;

    if polished.cost.is_finite() && polished.cost <= candidate_outcome.cost {
        Ok(polished)
    } else {
        Ok(candidate_outcome)
    }
}

/// Fit via the Nelder-Mead simplex method (derivative-free, local-ish) followed
/// by an LM polish. Cheaper than PSO but might not escape a bad basin.
pub fn nelder_mead_fit(
    topology: &[Node],
    frequencies: &[f64],
    z_measured: &[Complex64],
    weighting: Weighting,
    max_iterations: u64,
) -> Result<FitOutcome, FitError> {
    if frequencies.len() != z_measured.len() {
        return Err(FitError::LengthMismatch);
    }
    if frequencies.is_empty() {
        return Err(FitError::EmptyData);
    }
    if circuit::param_count(topology) == 0 {
        return Err(FitError::NoFreeParameters);
    }

    let omegas: Vec<f64> = frequencies.iter().map(|f| TAU * f).collect();
    let weights = compute_weights(z_measured, weighting);
    let bounds = circuit::param_bounds(topology);
    let guess = circuit::param_values(topology);
    let guess_coord: Vec<f64> = guess.iter().zip(&bounds).map(|(&p, &b)| to_pso_coord(p, b)).collect();

    // A simplex needs n+1 vertices for an n-dimensional problem: the guess itself,
    // plus one point per dimension nudged along that axis (10% of its own coordinate
    // magnitude, or a fixed small step if the coordinate is ~0).
    let n = guess_coord.len();
    let mut simplex = vec![guess_coord.clone()];
    for i in 0..n {
        let mut point = guess_coord.clone();
        point[i] += if point[i].abs() > 1e-8 { point[i].abs() * 0.1 } else { 0.1 };
        simplex.push(point);
    }

    let problem = PsoProblem { topology, omegas: &omegas, z_measured, weights: &weights, bounds: &bounds };
    let solver = NelderMead::new(simplex);

    let result = Executor::new(problem, solver)
        .configure(|state| state.max_iters(max_iterations))
        .run()
        .map_err(|e| FitError::SolverError(e.to_string()))?;

    let best_coord = result
        .state
        .get_best_param()
        .cloned()
        .ok_or_else(|| FitError::SolverError("nelder-mead found no best position".to_string()))?;
    let best: Vec<f64> = best_coord.iter().zip(&bounds).map(|(&c, &b)| from_pso_coord(c, b)).collect();
    let iterations = result.state.get_iter();

    polish_or_fallback(topology, best, iterations, frequencies, z_measured, weighting, &omegas, &weights)
}

/// Number of independent restarts for `differential_evolution_fit`.
const DE_RESTARTS: usize = 10;

/// Fit via self-adaptive differential evolution followed by LM polish.
/// The DE crate works in `f32`, LM recovers full `f64` precision.
pub fn differential_evolution_fit(
    topology: &[Node],
    frequencies: &[f64],
    z_measured: &[Complex64],
    weighting: Weighting,
    num_evaluations: usize,
) -> Result<FitOutcome, FitError> {
    if frequencies.len() != z_measured.len() {
        return Err(FitError::LengthMismatch);
    }
    if frequencies.is_empty() {
        return Err(FitError::EmptyData);
    }
    if circuit::param_count(topology) == 0 {
        return Err(FitError::NoFreeParameters);
    }

    let omegas: Vec<f64> = frequencies.iter().map(|f| TAU * f).collect();
    let weights = compute_weights(z_measured, weighting);
    let bounds = circuit::param_bounds(topology);
    let guess = circuit::param_values(topology);
    let (lower, upper) = pso_search_box(&bounds, &guess);
    let coord_bounds: Vec<(f32, f32)> =
        lower.iter().zip(&upper).map(|(&lo, &hi)| (lo as f32, hi as f32)).collect();

    // Selection across restarts is done with our own f64 residuals(), not the DE
    // crate's internal f32 cost. A solution with a wildly wrong parameter (e.g. R in
    // a nearly-flat cost direction pushed to an extreme value) can round to the same
    // or lower f32 cost as the true optimum, so we cannot trust the crate's own
    // f32 comparison to pick "best across restarts".
    let mut overall_best: Option<(f64, Vec<f64>)> = None;
    let mut total_evaluations = 0u64;

    for _ in 0..DE_RESTARTS {
        let topology_owned = topology.to_vec();
        let omegas_owned = omegas.clone();
        let z_owned = z_measured.to_vec();
        let weights_owned = weights.clone();
        let bounds_owned = bounds.clone();

        let mut de = self_adaptive_de(coord_bounds.clone(), move |coord: &[f32]| {
            let p: Vec<f64> =
                coord.iter().zip(&bounds_owned).map(|(&c, &b)| from_pso_coord(f64::from(c), b)).collect();
            let r = residuals(&topology_owned, &p, &omegas_owned, &z_owned, &weights_owned);
            (0.5 * r.iter().map(|x| x * x).sum::<f64>()) as f32
        });

        de.iter().nth(num_evaluations.max(1) - 1);
        total_evaluations += de.num_cost_evaluations() as u64;

        if let Some((_, coord)) = de.best() {
            let p: Vec<f64> = coord.iter().zip(&bounds).map(|(&c, &b)| from_pso_coord(f64::from(c), b)).collect();
            let r = residuals(topology, &p, &omegas, z_measured, &weights);
            let cost_f64 = 0.5 * r.iter().map(|x| x * x).sum::<f64>();
            if overall_best.as_ref().is_none_or(|(best_cost, _)| cost_f64 < *best_cost) {
                overall_best = Some((cost_f64, p));
            }
        }
    }

    let (_, best) = overall_best
        .ok_or_else(|| FitError::SolverError("differential evolution found no best position".to_string()))?;

    polish_or_fallback(topology, best, total_evaluations, frequencies, z_measured, weighting, &omegas, &weights)
}

/// `argmin::core::CostFunction` + `Anneal` adapter for simulated annealing. Shares
/// `PsoProblem`'s coordinate convention (log-space for open-ended parameters) but
/// needs its own struct since `Anneal::anneal` requires interior-mutable RNG state
/// (`&self`, not `&mut self`).
struct SaProblem<'a> {
    topology: &'a [Node],
    omegas: &'a [f64],
    z_measured: &'a [Complex64],
    weights: &'a [f64],
    bounds: &'a [(f64, f64)],
    rng: RefCell<rand::rngs::StdRng>,
}

impl CostFunction for SaProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, coord: &Vec<f64>) -> Result<f64, argmin::core::Error> {
        let p: Vec<f64> = coord.iter().zip(self.bounds).map(|(&c, &b)| from_pso_coord(c, b)).collect();
        let r = residuals(self.topology, &p, self.omegas, self.z_measured, self.weights);
        Ok(0.5 * r.iter().map(|x| x * x).sum::<f64>())
    }
}

impl Anneal for SaProblem<'_> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Float = f64;

    /// `extent` is the solver's current temperature (see argmin's SimulatedAnnealing
    /// source: `problem.anneal(&prev_param, self.cur_temp)`), so perturbing uniformly
    /// within `[-extent, extent]` per coordinate naturally shrinks the step size as
    /// the temperature cools -- the standard SA convention.
    fn anneal(&self, param: &Vec<f64>, extent: f64) -> Result<Vec<f64>, argmin::core::Error> {
        let mut rng = self.rng.borrow_mut();
        Ok(param.iter().map(|&c| c + rng.random_range(-extent..=extent)).collect())
    }
}

/// Fit via simulated annealing followed by an LM polish.
/// `seed`: pass `Some(seed)` for a reproducible run.
#[allow(clippy::too_many_arguments)]
pub fn simulated_annealing_fit(
    topology: &[Node],
    frequencies: &[f64],
    z_measured: &[Complex64],
    weighting: Weighting,
    max_iterations: u64,
    initial_temperature: f64,
    seed: Option<u64>,
) -> Result<FitOutcome, FitError> {
    if frequencies.len() != z_measured.len() {
        return Err(FitError::LengthMismatch);
    }
    if frequencies.is_empty() {
        return Err(FitError::EmptyData);
    }
    if circuit::param_count(topology) == 0 {
        return Err(FitError::NoFreeParameters);
    }

    let omegas: Vec<f64> = frequencies.iter().map(|f| TAU * f).collect();
    let weights = compute_weights(z_measured, weighting);
    let bounds = circuit::param_bounds(topology);
    let guess = circuit::param_values(topology);
    let guess_coord: Vec<f64> = guess.iter().zip(&bounds).map(|(&p, &b)| to_pso_coord(p, b)).collect();

    let anneal_rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_os_rng(),
    };
    let accept_rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s.wrapping_add(1)),
        None => rand::rngs::StdRng::from_os_rng(),
    };
    let problem = SaProblem {
        topology,
        omegas: &omegas,
        z_measured,
        weights: &weights,
        bounds: &bounds,
        rng: RefCell::new(anneal_rng),
    };
    let solver = SimulatedAnnealing::new_with_rng(initial_temperature, accept_rng)
        .map_err(|e| FitError::SolverError(e.to_string()))?;

    let result = Executor::new(problem, solver)
        .configure(|state| state.param(guess_coord).max_iters(max_iterations))
        .run()
        .map_err(|e| FitError::SolverError(e.to_string()))?;

    let best_coord = result
        .state
        .get_best_param()
        .cloned()
        .ok_or_else(|| FitError::SolverError("simulated annealing found no best position".to_string()))?;
    let best: Vec<f64> = best_coord.iter().zip(&bounds).map(|(&c, &b)| from_pso_coord(c, b)).collect();
    let iterations = result.state.get_iter();

    polish_or_fallback(topology, best, iterations, frequencies, z_measured, weighting, &omegas, &weights)
}

/// Fit via basin-hopping: repeated perturbation + LM.
/// `seed`: pass `Some(seed)` for a reproducible run.
#[allow(clippy::too_many_arguments)]
pub fn basin_hopping_fit(
    topology: &[Node],
    frequencies: &[f64],
    z_measured: &[Complex64],
    weighting: Weighting,
    num_hops: u32,
    step_size: f64,
    temperature: f64,
    seed: Option<u64>,
) -> Result<FitOutcome, FitError> {
    if frequencies.len() != z_measured.len() {
        return Err(FitError::LengthMismatch);
    }
    if frequencies.is_empty() {
        return Err(FitError::EmptyData);
    }
    if circuit::param_count(topology) == 0 {
        return Err(FitError::NoFreeParameters);
    }

    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_os_rng(),
    };
    let bounds = circuit::param_bounds(topology);

    // Step 0: a full local LM fit from the caller's own initial guess, exactly like
    // scipy.optimize.basinhopping's first step.
    let mut current =
        levenberg_marquardt_fit(topology, frequencies, z_measured, weighting, &FitOptions::default())?;
    let mut best = current.params.clone();
    let mut best_cost = current.cost;
    let mut total_iterations = current.iterations;

    for _ in 0..num_hops {
        let current_coord: Vec<f64> =
            current.params.iter().zip(&bounds).map(|(&p, &b)| to_pso_coord(p, b)).collect();
        let proposal_coord: Vec<f64> =
            current_coord.iter().map(|&c| c + rng.random_range(-step_size..=step_size)).collect();
        let proposal: Vec<f64> =
            proposal_coord.iter().zip(&bounds).map(|(&c, &b)| from_pso_coord(c, b)).collect();

        let proposal_topology = circuit::with_param_values(topology, &proposal);
        let candidate =
            levenberg_marquardt_fit(&proposal_topology, frequencies, z_measured, weighting, &FitOptions::default())?;
        total_iterations += candidate.iterations;

        if candidate.cost.is_finite() && candidate.cost < best_cost {
            best = candidate.params.clone();
            best_cost = candidate.cost;
        }

        let accept = candidate.cost.is_finite()
            && (candidate.cost < current.cost
                || rng.random::<f64>() < (-(candidate.cost - current.cost) / temperature).exp());
        if accept {
            current = candidate;
        }
    }

    let omegas: Vec<f64> = frequencies.iter().map(|f| TAU * f).collect();
    let weights = compute_weights(z_measured, weighting);
    Ok(build_outcome(topology, best, true, total_iterations, &omegas, z_measured, &weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::Element;

    fn r(value: f64) -> Node {
        Node::Element(Element::R { r: value }, None)
    }

    fn cpe(q: f64, alpha: f64) -> Node {
        Node::Element(Element::Cpe { q, alpha }, None)
    }

    fn zarc(r: f64, tau_k: f64, gamma: f64) -> Series {
        vec![Node::Element(Element::Zarc { r, tau_k, gamma }, None)]
    }

    fn synthetic_data(truth: &[Node], frequencies: &[f64]) -> Vec<Complex64> {
        frequencies.iter().map(|f| circuit::impedance(truth, TAU * f)).collect()
    }

    fn log_spaced_freqs(low: f64, high: f64, n: usize) -> Vec<f64> {
        let log_low = low.ln();
        let log_high = high.ln();
        (0..n)
            .map(|i| (log_low + (log_high - log_low) * i as f64 / (n - 1) as f64).exp())
            .collect()
    }

    #[test]
    fn recovers_single_resistor_from_noise_free_data() {
        let truth = vec![r(100.0)];
        let guess = vec![r(50.0)];
        let freqs = log_spaced_freqs(1.0, 1e5, 20);
        let z = synthetic_data(&truth, &freqs);

        let outcome =
            levenberg_marquardt_fit(&guess, &freqs, &z, Weighting::Modulus, &FitOptions::default()).unwrap();

        assert!(outcome.success);
        assert!((outcome.params[0] - 100.0).abs() < 1e-6, "params={:?}", outcome.params);
    }

    #[test]
    fn recovers_randles_cell_from_noise_free_synthetic_data() {
        // Rs - p(Rct, W) - Cdl-style Randles cell, mirroring the Python test fixtures.
        let truth = vec![
            r(20.0),
            Node::Parallel(vec![vec![r(200.0)], vec![Node::Element(Element::W { aw: 50.0 }, None)]]),
            Node::Element(Element::C { c: 1e-5 }, None),
        ];
        let guess = vec![
            r(25.0),
            Node::Parallel(vec![vec![r(150.0)], vec![Node::Element(Element::W { aw: 65.0 }, None)]]),
            Node::Element(Element::C { c: 1.3e-5 }, None),
        ];
        let freqs = log_spaced_freqs(0.1, 1e5, 50);
        let z = synthetic_data(&truth, &freqs);

        let outcome =
            levenberg_marquardt_fit(&guess, &freqs, &z, Weighting::Modulus, &FitOptions::default()).unwrap();

        assert!(outcome.success);
        let expected = circuit::param_values(&truth);
        for (fitted, exp) in outcome.params.iter().zip(&expected) {
            let rel_err = (fitted - exp).abs() / exp.abs();
            assert!(rel_err < 1e-3, "fitted={:?} expected={:?}", outcome.params, expected);
        }
    }

    #[test]
    fn modulus_weighting_differs_from_unit_weighting_on_multiscale_data() {
        // Two elements whose impedance contributions differ by several orders of
        // magnitude across the sweep -- unweighted LS should be dominated by the
        // low-frequency (large-|Z|) end, giving a different optimum than modulus weighting.
        let truth = vec![r(1.0), cpe(1e-2, 0.7)];
        let guess = vec![r(3.0), cpe(5e-3, 0.5)];
        let freqs = log_spaced_freqs(1.0, 1e6, 40);
        let mut z = synthetic_data(&truth, &freqs);
        // Light deterministic perturbation so the two weightings aren't trivially identical.
        for (i, zi) in z.iter_mut().enumerate() {
            let bump = 1.0 + 0.01 * ((i as f64) * 0.7).sin();
            *zi *= bump;
        }

        let modulus =
            levenberg_marquardt_fit(&guess, &freqs, &z, Weighting::Modulus, &FitOptions::default()).unwrap();
        let unit = levenberg_marquardt_fit(&guess, &freqs, &z, Weighting::Unit, &FitOptions::default()).unwrap();

        let differ = modulus.params.iter().zip(&unit.params).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differ, "modulus={:?} unit={:?}", modulus.params, unit.params);
    }

    #[test]
    fn respects_bounds_and_never_returns_out_of_range_values() {
        // Adversarial guess/data pulling a Zarc gamma toward/past the [0,1] boundary.
        let truth = zarc(50.0, 0.2, 0.95);
        let guess = zarc(10.0, 0.05, 0.3);
        let freqs = log_spaced_freqs(1.0, 1e6, 30);
        let z = synthetic_data(&truth, &freqs);

        let outcome =
            levenberg_marquardt_fit(&guess, &freqs, &z, Weighting::Modulus, &FitOptions::default()).unwrap();

        let bounds = circuit::param_bounds(&guess);
        for (&value, &(lo, hi)) in outcome.params.iter().zip(&bounds) {
            assert!(value >= lo && value <= hi, "value {value} outside [{lo}, {hi}]");
        }
    }

    #[test]
    fn particle_swarm_converges_on_a_hard_multi_branch_topology() {
        // Two correlated R//CPE branches plus a series inductor, fit from a rough
        // initial guess. A multi-modal landscape that needs a global optimiser to
        // solve. For PSO it is possible but dependends heavily on the seed.
        let truth = vec![
            Node::Element(Element::L { l: 1e-7 }, None),
            r(0.05),
            Node::Parallel(vec![vec![r(0.02)], vec![cpe(0.5, 0.85)]]),
            Node::Parallel(vec![vec![r(0.08)], vec![cpe(0.05, 0.7)]]),
        ];
        let guess = vec![
            Node::Element(Element::L { l: 1e-8 }, None),
            r(0.03),
            Node::Parallel(vec![vec![r(0.01)], vec![cpe(1.0, 1.0)]]),
            Node::Parallel(vec![vec![r(0.05)], vec![cpe(1.0, 1.0)]]),
        ];
        let freqs = log_spaced_freqs(1e-1, 1e5, 40);
        let z = synthetic_data(&truth, &freqs);

        let pso = particle_swarm_fit(&guess, &freqs, &z, Weighting::Modulus, 200, 1000, Some(1729)).unwrap();

        assert!(pso.success);
        // R//CPE branches are notoriously non-identifiable, so this doesn't demand
        // recovering the exact parameters, just a good fit
        assert!(pso.cost < 5e-3, "pso.cost={} too high, params={:?}", pso.cost, pso.params);
    }

    /// Same Randles-cell scenario as `recovers_randles_cell_from_noise_free_synthetic_data`,
    /// reused across the new solvers below so each test is just "does this method
    /// actually converge on a moderately hard, realistic topology".
    fn randles_case() -> (Series, Series, Vec<f64>, Vec<Complex64>) {
        let truth = vec![
            r(20.0),
            Node::Parallel(vec![vec![r(200.0)], vec![Node::Element(Element::W { aw: 50.0 }, None)]]),
            Node::Element(Element::C { c: 1e-5 }, None),
        ];
        let guess = vec![
            r(25.0),
            Node::Parallel(vec![vec![r(150.0)], vec![Node::Element(Element::W { aw: 65.0 }, None)]]),
            Node::Element(Element::C { c: 1.3e-5 }, None),
        ];
        let freqs = log_spaced_freqs(0.1, 1e5, 50);
        let z = synthetic_data(&truth, &freqs);
        (truth, guess, freqs, z)
    }

    /// Checks impedance-space agreement. R1 turns out to sit in a nearly-flat 
    /// cost direction across this frequency range specifically --
    /// R0/W/C all converge tightly for every method here, but a global search can
    /// occasionally drift R1 to an extreme value while still reproducing
    /// the impedance curve well, since 1/R1 -> 0 just means that branch degenerates
    /// toward the Warburg element alone. Does not mean a bad fit.
    fn assert_recovers_randles(freqs: &[f64], z_measured: &[Complex64], outcome: &FitOutcome) {
        assert!(outcome.success);
        for (&f, &z) in freqs.iter().zip(z_measured) {
            let model = circuit::impedance(&outcome.node, TAU * f);
            let rel_err = (model - z).norm() / z.norm();
            assert!(rel_err < 1e-2, "f={f} model={model:?} measured={z:?} params={:?}", outcome.params);
        }
    }

    #[test]
    fn nelder_mead_recovers_randles_cell() {
        let (_truth, guess, freqs, z) = randles_case();
        let outcome = nelder_mead_fit(&guess, &freqs, &z, Weighting::Modulus, 2000).unwrap();
        assert_recovers_randles(&freqs, &z, &outcome);
    }

    #[test]
    fn differential_evolution_recovers_randles_cell() {
        let (_truth, guess, freqs, z) = randles_case();
        let outcome = differential_evolution_fit(&guess, &freqs, &z, Weighting::Modulus, 20_000).unwrap();
        assert_recovers_randles(&freqs, &z, &outcome);
    }

    #[test]
    fn simulated_annealing_recovers_randles_cell() {
        let (_truth, guess, freqs, z) = randles_case();
        let outcome =
            simulated_annealing_fit(&guess, &freqs, &z, Weighting::Modulus, 5000, 2.0, Some(1729)).unwrap();
        assert_recovers_randles(&freqs, &z, &outcome);
    }

    #[test]
    fn basin_hopping_recovers_randles_cell() {
        let (_truth, guess, freqs, z) = randles_case();
        let outcome = basin_hopping_fit(&guess, &freqs, &z, Weighting::Modulus, 20, 1.0, 1.0, Some(1729)).unwrap();
        assert_recovers_randles(&freqs, &z, &outcome);
    }

    #[test]
    fn new_solvers_reject_mismatched_or_empty_input() {
        let topo = vec![r(100.0)];
        let freqs = vec![1.0, 2.0];
        let z = vec![Complex64::new(1.0, 0.0)];
        assert!(matches!(
            nelder_mead_fit(&topo, &freqs, &z, Weighting::Unit, 100),
            Err(FitError::LengthMismatch)
        ));
        assert!(matches!(
            differential_evolution_fit(&topo, &freqs, &z, Weighting::Unit, 100),
            Err(FitError::LengthMismatch)
        ));
        assert!(matches!(
            simulated_annealing_fit(&topo, &freqs, &z, Weighting::Unit, 100, 1.0, Some(1)),
            Err(FitError::LengthMismatch)
        ));
        assert!(matches!(
            basin_hopping_fit(&topo, &freqs, &z, Weighting::Unit, 5, 1.0, 1.0, Some(1)),
            Err(FitError::LengthMismatch)
        ));
    }

    #[test]
    fn converges_cleanly_when_initial_guess_sits_exactly_on_a_bound() {
        // LM should still convertge when initial guess is exactly on a bound.
        // Regression test, as a CPE alpha starting at 1.0 used to break the fit
        // due to a particular bound clamping method.
        let truth = Node::Element(Element::Cpe { q: 1e-2, alpha: 0.85 }, None);
        let guess = Node::Element(Element::Cpe { q: 1.0, alpha: 1.0 }, None);
        let freqs = log_spaced_freqs(1.0, 1e5, 30);
        let z = synthetic_data(&[truth], &freqs);

        let outcome =
            levenberg_marquardt_fit(&[guess], &freqs, &z, Weighting::Modulus, &FitOptions::default()).unwrap();

        assert!(outcome.success);
        assert!(
            (outcome.params[1] - 0.85).abs() < 1e-2,
            "alpha should not be pinned to a boundary: params={:?}",
            outcome.params
        );
        let bounds = circuit::param_bounds(&[Node::Element(Element::Cpe { q: 1.0, alpha: 1.0 }, None)]);
        for (&v, &(lo, hi)) in outcome.params.iter().zip(&bounds) {
            assert!(v > lo && v < hi, "param {v} not strictly inside ({lo}, {hi})");
        }
    }

    #[test]
    fn returns_success_false_but_still_usable_when_max_iterations_too_small() {
        let truth =
            vec![r(20.0), Node::Parallel(vec![vec![r(200.0)], vec![Node::Element(Element::W { aw: 50.0 }, None)]])];
        let guess =
            vec![r(80.0), Node::Parallel(vec![vec![r(20.0)], vec![Node::Element(Element::W { aw: 5.0 }, None)]])];
        let freqs = log_spaced_freqs(0.1, 1e5, 40);
        let z = synthetic_data(&truth, &freqs);

        let options = FitOptions { max_iterations: 1, ..FitOptions::default() };
        let outcome = levenberg_marquardt_fit(&guess, &freqs, &z, Weighting::Modulus, &options).unwrap();

        for &v in &outcome.params {
            assert!(v.is_finite());
        }
        assert_eq!(outcome.params.len(), 3);
    }

    #[test]
    fn handles_rank_deficient_jacobian_without_panicking() {
        // Far more parameters than independent data points -> JTJ is singular.
        let truth = vec![r(10.0), r(20.0), r(30.0), r(40.0), r(50.0)];
        let guess = vec![r(11.0), r(21.0), r(31.0), r(41.0), r(51.0)];
        let freqs = vec![1.0, 10.0];
        let z = synthetic_data(&truth, &freqs);

        let outcome =
            levenberg_marquardt_fit(&guess, &freqs, &z, Weighting::Modulus, &FitOptions::default()).unwrap();

        assert!(outcome.stderr.is_none());
        assert_eq!(outcome.params.len(), 5);
    }

    #[test]
    fn rejects_mismatched_or_empty_input() {
        let topo = vec![r(100.0)];
        assert!(matches!(
            levenberg_marquardt_fit(&topo, &[1.0, 2.0], &[Complex64::new(1.0, 0.0)], Weighting::Unit, &FitOptions::default()),
            Err(FitError::LengthMismatch)
        ));
        assert!(matches!(
            levenberg_marquardt_fit(&topo, &[], &[], Weighting::Unit, &FitOptions::default()),
            Err(FitError::EmptyData)
        ));
    }

    /// Check fasteis finite-difference Jacobian against the levenberg-marquardt
    /// crate's own (much slower, debug-oriented) numerical differentiation helper.
    #[test]
    fn jacobian_matches_crate_numerical_differentiation() {
        // Use only simple elements (R/L) so this is a check of 
        // jacobian_columns/DMatrix-assembly mechanics.
        let topology =
            vec![r(20.0), Node::Parallel(vec![vec![r(200.0)], vec![Node::Element(Element::L { l: 0.05 }, None)]])];
        let freqs = log_spaced_freqs(1.0, 1e5, 15);
        let omegas: Vec<f64> = freqs.iter().map(|f| TAU * f).collect();
        let z_measured: Vec<Complex64> = freqs.iter().map(|f| circuit::impedance(&topology, TAU * f)).collect();
        let weights = compute_weights(&z_measured, Weighting::Modulus);
        let bounds = circuit::param_bounds(&topology);
        let p0 = circuit::param_values(&topology);
        let coord = DVector::from_vec(p0.iter().zip(&bounds).map(|(&p, &b)| to_pso_coord(p, b)).collect());
        let fixed = vec![false; p0.len()];

        let mut problem =
            LmProblem { topology: &topology, omegas, z_measured: &z_measured, weights, bounds, coord, fixed };

        let ours = problem.jacobian().unwrap();
        let numerical = levenberg_marquardt::differentiate_numerically(&mut problem).unwrap();

        let max_rel_err = ours
            .iter()
            .zip(numerical.iter())
            .map(|(a, b)| (a - b).abs() / b.abs().max(1e-8))
            .fold(0.0, f64::max);
        assert!(max_rel_err < 1e-4, "max_rel_err={max_rel_err} ours={ours:?} numerical={numerical:?}");
    }
}

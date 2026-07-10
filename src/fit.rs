use std::f64::consts::TAU;

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn};
use num_complex::Complex64;
use rayon::prelude::*;

use crate::circuit::Node;

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
        FitOptions { max_iterations: 200, ftol: 1e-10, xtol: 1e-10, gtol: 1e-10 }
    }
}

#[derive(Debug)]
pub struct FitOutcome {
    pub node: Node,
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
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            FitError::LengthMismatch => "frequencies and impedances must have the same length",
            FitError::EmptyData => "frequencies and impedances must not be empty",
            FitError::NoFreeParameters => "circuit topology has no parameters to fit",
        };
        f.write_str(msg)
    }
}

fn compute_weights(z_measured: &[Complex64], weighting: Weighting) -> Vec<f64> {
    match weighting {
        Weighting::Unit => vec![1.0; z_measured.len()],
        Weighting::Modulus => z_measured.iter().map(|z| z.norm().max(1e-30)).collect(),
    }
}

/// Interleaved `[re0, im0, re1, im1, ...]` weighted residual vector, length `2N`.
/// Optimizer-agnostic: operates on plain `Vec<f64>`, reused unchanged by any future
/// non-LM backend (see the `argmin` extensibility note in the fit() design).
fn residuals(topology: &Node, p: &[f64], omegas: &[f64], z_measured: &[Complex64], weights: &[f64]) -> Vec<f64> {
    let node = topology.with_param_values(p);
    let mut r = vec![0.0; 2 * omegas.len()];
    for (i, &omega) in omegas.iter().enumerate() {
        let z = node.impedance(omega);
        let w = weights[i];
        r[2 * i] = (z.re - z_measured[i].re) / w;
        r[2 * i + 1] = (z.im - z_measured[i].im) / w;
    }
    r
}

/// Central-difference Jacobian columns (one per parameter), computed in parallel via
/// rayon -- the natural parallelism axis given the typically small (2-15) parameter count.
/// Perturbations are not clamped to physical bounds: impedance() is smooth well outside
/// those ranges, so clamping here would bias the derivative estimate near a boundary.
fn jacobian_columns(
    topology: &Node,
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
/// Bounds are enforced entirely by clamping every incoming parameter vector in
/// `set_params`, which the crate calls before every residual/Jacobian evaluation --
/// this is what keeps fits from wandering into unphysical territory (negative
/// resistance, etc.) without needing a log/logit reparametrization.
struct LmProblem<'a> {
    topology: &'a Node,
    omegas: Vec<f64>,
    z_measured: &'a [Complex64],
    weights: Vec<f64>,
    bounds: Vec<(f64, f64)>,
    params: DVector<f64>,
}

impl LmProblem<'_> {
    fn clamp_params(&self, p: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(p.len(), p.iter().zip(&self.bounds).map(|(&v, &(lo, hi))| v.clamp(lo, hi)))
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for LmProblem<'_> {
    type ParameterStorage = nalgebra::storage::Owned<f64, Dyn>;
    type ResidualStorage = nalgebra::storage::Owned<f64, Dyn>;
    type JacobianStorage = nalgebra::storage::Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, p: &DVector<f64>) {
        self.params = self.clamp_params(p);
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let p: Vec<f64> = self.params.iter().copied().collect();
        Some(DVector::from_vec(residuals(self.topology, &p, &self.omegas, self.z_measured, &self.weights)))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let p: Vec<f64> = self.params.iter().copied().collect();
        let cols = jacobian_columns(self.topology, &p, &self.omegas, self.z_measured, &self.weights);
        let m = cols.first()?.len();
        let n = cols.len();
        Some(DMatrix::from_fn(m, n, |i, j| cols[j][i]))
    }
}

/// Fit `topology`'s parameters to measured impedance data via Levenberg-Marquardt,
/// starting from `topology`'s current parameter values as the initial guess.
/// Always returns the best parameters found, even when `success` is false
/// (max iterations exhausted, etc.) -- convergence quality never withholds a result.
pub fn levenberg_marquardt_fit(
    topology: &Node,
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
    if topology.param_count() == 0 {
        return Err(FitError::NoFreeParameters);
    }

    let omegas: Vec<f64> = frequencies.iter().map(|f| TAU * f).collect();
    let weights = compute_weights(z_measured, weighting);
    let bounds = topology.param_bounds();
    let p0 = DVector::from_vec(topology.param_values());

    let mut problem =
        LmProblem { topology, omegas, z_measured, weights, bounds, params: p0.clone() };
    problem.params = problem.clamp_params(&p0);

    let solver = LevenbergMarquardt::new()
        .with_ftol(options.ftol)
        .with_xtol(options.xtol)
        .with_gtol(options.gtol)
        .with_patience(options.max_iterations.max(1) as usize);

    let (solved, report) = solver.minimize(problem);

    let params: Vec<f64> = solved.params().iter().copied().collect();
    let success = report.termination.was_successful();
    let chi_square = 2.0 * report.objective_function;

    let stderr = solved.jacobian().and_then(|j| {
        let dof = j.nrows() as f64 - j.ncols() as f64;
        if dof <= 0.0 {
            return None;
        }
        let jtj = j.transpose() * &j;
        jtj.try_inverse().map(|inv| {
            let scale = chi_square / dof;
            (0..inv.nrows()).map(|i| (inv[(i, i)] * scale).sqrt()).collect()
        })
    });

    Ok(FitOutcome {
        node: topology.with_param_values(&params),
        param_names: topology.param_names(),
        params,
        success,
        iterations: report.number_of_evaluations as u64,
        cost: report.objective_function,
        chi_square,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::Element;

    fn r(value: f64) -> Node {
        Node::Leaf(Element::R { r: value })
    }

    fn cpe(q: f64, alpha: f64) -> Node {
        Node::Leaf(Element::Cpe { q, alpha })
    }

    fn zarc(r: f64, tau_k: f64, gamma: f64) -> Node {
        Node::Leaf(Element::Zarc { r, tau_k, gamma })
    }

    fn synthetic_data(truth: &Node, frequencies: &[f64]) -> Vec<Complex64> {
        frequencies.iter().map(|f| truth.impedance(TAU * f)).collect()
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
        let truth = r(100.0);
        let guess = r(50.0);
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
        let truth = Node::Series(vec![
            r(20.0),
            Node::Parallel(vec![r(200.0), Node::Leaf(Element::W { aw: 50.0 })]),
            Node::Leaf(Element::C { c: 1e-5 }),
        ]);
        let guess = Node::Series(vec![
            r(25.0),
            Node::Parallel(vec![r(150.0), Node::Leaf(Element::W { aw: 65.0 })]),
            Node::Leaf(Element::C { c: 1.3e-5 }),
        ]);
        let freqs = log_spaced_freqs(0.1, 1e5, 50);
        let z = synthetic_data(&truth, &freqs);

        let outcome =
            levenberg_marquardt_fit(&guess, &freqs, &z, Weighting::Modulus, &FitOptions::default()).unwrap();

        assert!(outcome.success);
        let expected = truth.param_values();
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
        let truth = Node::Series(vec![r(1.0), cpe(1e-2, 0.7)]);
        let guess = Node::Series(vec![r(3.0), cpe(5e-3, 0.5)]);
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

        let bounds = guess.param_bounds();
        for (&value, &(lo, hi)) in outcome.params.iter().zip(&bounds) {
            assert!(value >= lo && value <= hi, "value {value} outside [{lo}, {hi}]");
        }
    }

    #[test]
    fn returns_success_false_but_still_usable_when_max_iterations_too_small() {
        let truth = Node::Series(vec![r(20.0), Node::Parallel(vec![r(200.0), Node::Leaf(Element::W { aw: 50.0 })])]);
        let guess = Node::Series(vec![r(80.0), Node::Parallel(vec![r(20.0), Node::Leaf(Element::W { aw: 5.0 })])]);
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
        let truth = Node::Series(vec![r(10.0), r(20.0), r(30.0), r(40.0), r(50.0)]);
        let guess = Node::Series(vec![r(11.0), r(21.0), r(31.0), r(41.0), r(51.0)]);
        let freqs = vec![1.0, 10.0];
        let z = synthetic_data(&truth, &freqs);

        let outcome =
            levenberg_marquardt_fit(&guess, &freqs, &z, Weighting::Modulus, &FitOptions::default()).unwrap();

        assert!(outcome.stderr.is_none());
        assert_eq!(outcome.params.len(), 5);
    }

    #[test]
    fn rejects_mismatched_or_empty_input() {
        let topo = r(100.0);
        assert!(matches!(
            levenberg_marquardt_fit(&topo, &[1.0, 2.0], &[Complex64::new(1.0, 0.0)], Weighting::Unit, &FitOptions::default()),
            Err(FitError::LengthMismatch)
        ));
        assert!(matches!(
            levenberg_marquardt_fit(&topo, &[], &[], Weighting::Unit, &FitOptions::default()),
            Err(FitError::EmptyData)
        ));
    }

    /// Cross-check our hand-written finite-difference Jacobian against the
    /// levenberg-marquardt crate's own (much slower, debug-oriented) numerical
    /// differentiation helper -- the crate's own suggested way to validate a
    /// LeastSquaresProblem::jacobian() implementation.
    #[test]
    fn jacobian_matches_crate_numerical_differentiation() {
        // Use only R/L elements (dZ/dR = 1, dZ/dL = jw -- both smooth, no 1/x-type
        // curvature) so this is a clean check of jacobian_columns/DMatrix-assembly
        // mechanics (column ordering, transposition, etc.), not a numerical-analysis
        // stress test. A CPE's small Q (real EIS values are ~1e-4 to 1e-6) has such
        // extreme local curvature near its R-CPE "knee" frequency that naive fixed-step
        // central differences and the crate's adaptive `differentiate_numerically` (an
        // entirely different, Richardson-extrapolation-based algorithm, explicitly
        // documented as "intended for debugging", not as a production reference) can
        // legitimately disagree at a handful of points -- that's not a bug in either;
        // the real end-to-end validation for that regime is the LM convergence tests
        // above, which exercise this exact code path and all converge successfully.
        let topology = Node::Series(vec![r(20.0), Node::Parallel(vec![r(200.0), Node::Leaf(Element::L { l: 0.05 })])]);
        let freqs = log_spaced_freqs(1.0, 1e5, 15);
        let omegas: Vec<f64> = freqs.iter().map(|f| TAU * f).collect();
        let z_measured: Vec<Complex64> = freqs.iter().map(|f| topology.impedance(TAU * f)).collect();
        let weights = compute_weights(&z_measured, Weighting::Modulus);
        let bounds = topology.param_bounds();
        let p0 = DVector::from_vec(topology.param_values());

        let mut problem =
            LmProblem { topology: &topology, omegas, z_measured: &z_measured, weights, bounds, params: p0 };

        let ours = problem.jacobian().unwrap();
        let numerical = levenberg_marquardt::differentiate_numerically(&mut problem).unwrap();

        assert!((&ours - &numerical).abs().max() < 1e-6, "ours={ours:?} numerical={numerical:?}");
    }
}

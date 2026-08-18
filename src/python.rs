use std::collections::HashMap;
use std::f64::consts::TAU;
use std::ffi::CString;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyUserWarning, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::circuit::{self, Node, Series};
use crate::elements::Element;
use crate::fit::{self, FitOptions, Weighting};
use crate::models;

/// Look up the model trained for `node`'s topology and run it over the spectrum.
fn guess_params(
    node: &[Node],
    frequencies: &[f64],
    impedances: &[Complex64],
    weights: Option<&str>,
) -> PyResult<Vec<f64>> {
    let (guesser, permutation, expanded) = match weights {
        Some(path) => {
            let found = models::load_external(path, node)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            (found.model, found.permutation, found.expanded)
        }
        None => {
            let found = models::find_for_topology(node)
                .ok_or_else(|| PyValueError::new_err(models::describe_missing()))?;
            let guesser = found
                .model
                .guesser()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            (guesser, found.permutation, found.expanded)
        }
    };
    let values = guesser
        .guess(frequencies, impedances)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let values = models::apply_permutation(&permutation, &values);
    Ok(if expanded {
        circuit::contract_arc_values(node, &values)
    } else {
        values
    })
}

/// Raise a Python `UserWarning`
fn warn(py: Python<'_>, message: &str) -> PyResult<()> {
    let text = CString::new(message)
        .map_err(|_| PyValueError::new_err("warning message contained an interior nul byte"))?;
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), &text, 1)
}

/// Below this many frequency points, rayon's ~20 us overhead outweighs
/// the benefit of parallelizing, so we fall back to a plain sequential map.
/// Depends on circuit complexity, R breaks even at 12_000, randles at 1_500,
/// more complex would be lower.
const PARALLEL_THRESHOLD: usize = 1_000;

/// Accepted spellings of each battery data format column
const FREQUENCY_COLUMNS: [&str; 2] = ["Frequency / Hz", "frequency_hertz"];
const REAL_COLUMNS: [&str; 2] = ["Real Impedance / ohm", "real_impedance_ohm"];
const IMAGINARY_COLUMNS: [&str; 2] = ["Imaginary Impedance / ohm", "imaginary_impedance_ohm"];
const ABSOLUTE_COLUMNS: [&str; 2] = ["Absolute Impedance / ohm", "absolute_impedance_ohm"];
const PHASE_COLUMNS: [&str; 2] = ["Phase / deg", "phase_degree"];

/// `"a" or "b"`, for listing accepted spellings in an error message.
fn either_of(names: &[&str; 2]) -> String {
    format!("{:?} or {:?}", names[0], names[1])
}

/// Read a column by any of its accepted spellings, `None` if the frame has
/// neither. Anything without string keys (a list, an array) also yields `None`.
fn column(frame: &Bound<'_, PyAny>, names: &[&str; 2]) -> PyResult<Option<Vec<f64>>> {
    for name in names {
        let Ok(values) = frame.get_item(name) else {
            continue;
        };
        return values
            .extract::<Vec<f64>>()
            .map(Some)
            .map_err(|_| PyValueError::new_err(format!("column {name:?} is not numeric")));
    }
    Ok(None)
}

/// Read f, Z from a battery data format dataframe.
/// Skip rows were f is not positive.
fn unpack_frame(frame: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<Complex64>)> {
    let Some(frequencies) = column(frame, &FREQUENCY_COLUMNS)? else {
        return Err(PyValueError::new_err(format!(
            "expected a dataframe with a {} column, \
             or frequencies and impedances as two arguments",
            either_of(&FREQUENCY_COLUMNS)
        )));
    };

    let cartesian = (
        column(frame, &REAL_COLUMNS)?,
        column(frame, &IMAGINARY_COLUMNS)?,
    );
    let polar = (
        column(frame, &ABSOLUTE_COLUMNS)?,
        column(frame, &PHASE_COLUMNS)?,
    );
    let (impedances, columns) = match (cartesian, polar) {
        ((Some(real), Some(imaginary)), _) => {
            if real.len() != imaginary.len() {
                return Err(PyValueError::new_err(
                    "real and imaginary impedance columns must have the same length",
                ));
            }
            let z: Vec<Complex64> = real
                .iter()
                .zip(&imaginary)
                .map(|(re, im)| Complex64::new(*re, *im))
                .collect();
            (z, (&REAL_COLUMNS, &IMAGINARY_COLUMNS))
        }
        (_, (Some(absolute), Some(phase))) => {
            if absolute.len() != phase.len() {
                return Err(PyValueError::new_err(
                    "absolute impedance and phase columns must have the same length",
                ));
            }
            let z = absolute
                .iter()
                .zip(&phase)
                .map(|(r, deg)| Complex64::from_polar(*r, deg.to_radians()))
                .collect();
            (z, (&ABSOLUTE_COLUMNS, &PHASE_COLUMNS))
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "no impedance columns: expected {} together with {}, or {} together with {}",
                either_of(&REAL_COLUMNS),
                either_of(&IMAGINARY_COLUMNS),
                either_of(&ABSOLUTE_COLUMNS),
                either_of(&PHASE_COLUMNS),
            )));
        }
    };

    if frequencies.len() != impedances.len() {
        return Err(PyValueError::new_err(format!(
            "frequency column has {} rows but {} and {} have {}",
            frequencies.len(),
            either_of(columns.0),
            either_of(columns.1),
            impedances.len(),
        )));
    }

    // Only keep rows where frequency is positive.
    let (frequencies, impedances): (Vec<f64>, Vec<Complex64>) = frequencies
        .into_iter()
        .zip(impedances)
        .filter(|(f, _)| *f > 0.0)
        .unzip();
    if frequencies.is_empty() {
        return Err(PyValueError::new_err(format!(
            "no measured points: every row of the {} column is zero or negative",
            either_of(&FREQUENCY_COLUMNS)
        )));
    }
    Ok((frequencies, impedances))
}

/// Resolve a spectrum given either two sequences or a single dataframe.
/// Dataframes remove non-positive frequencies.
fn spectrum(
    frequencies: &Bound<'_, PyAny>,
    impedances: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<f64>, Vec<Complex64>)> {
    let Some(impedances) = impedances else {
        return unpack_frame(frequencies);
    };
    let pair = (
        frequencies.extract::<Vec<f64>>()?,
        impedances.extract::<Vec<Complex64>>()?,
    );
    if pair.0.len() != pair.1.len() {
        return Err(PyValueError::new_err(
            "frequencies and impedances must have the same length",
        ));
    }
    Ok(pair)
}

/// Resolve frequencies given either a sequence or a dataframe to take the
/// frequency column from.
fn frequencies_of(frequencies: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(values) = frequencies.extract::<Vec<f64>>() {
        return Ok(values);
    }
    column(frequencies, &FREQUENCY_COLUMNS)?.ok_or_else(|| {
        PyValueError::new_err(format!(
            "expected a sequence of frequencies, or a dataframe with a {} column",
            either_of(&FREQUENCY_COLUMNS)
        ))
    })
}

#[pyclass(module = "fasteis")]
#[derive(Clone)]
pub struct Circuit {
    pub node: Series,
    /// Whether user supplied initial values
    values_supplied: bool,
}

impl Circuit {
    /// A circuit whose parameter values came from the caller or from a fit.
    fn valued(node: Series) -> Circuit {
        Circuit {
            node,
            values_supplied: true,
        }
    }
}

fn parse_weighting(weight: &str) -> PyResult<Weighting> {
    match weight {
        "modulus" => Ok(Weighting::Modulus),
        "unit" => Ok(Weighting::Unit),
        other => Err(PyValueError::new_err(format!(
            "unknown weight scheme {other:?}, expected \"modulus\" or \"unit\""
        ))),
    }
}

/// A child accepted by `Series()` and `Parallel()`: either a single element,
/// or nested circuit.
#[derive(FromPyObject)]
enum Part {
    Element(Element),
    Circuit(Circuit),
}

impl Part {
    fn values_supplied(&self) -> bool {
        match self {
            Part::Element(_) => true,
            Part::Circuit(circuit) => circuit.values_supplied,
        }
    }

    fn into_series(self) -> Series {
        match self {
            Part::Element(element) => vec![Node::Element(element, None)],
            Part::Circuit(circuit) => circuit.node,
        }
    }
}

/// Connect elements and circuits end to end.
#[pyfunction]
#[pyo3(name = "Series")]
pub fn series(parts: Vec<Part>) -> Circuit {
    Circuit {
        values_supplied: parts.iter().all(Part::values_supplied),
        node: parts.into_iter().flat_map(Part::into_series).collect(),
    }
}

/// Connect elements and circuits as parallel branches.
#[pyfunction]
#[pyo3(name = "Parallel")]
pub fn parallel(parts: Vec<Part>) -> Circuit {
    Circuit {
        values_supplied: parts.iter().all(Part::values_supplied),
        node: vec![Node::Parallel(
            parts.into_iter().map(Part::into_series).collect(),
        )],
    }
}

#[pymethods]
impl Circuit {
    /// Parse a circuit topology string, e.g. `"R0-p(R1,Cpe1)"` or
    /// `"R0-p(R1-C1,R2-Cpe2)"`. The string carries no parameter values --
    /// every element gets a placeholder default; set real values afterward
    /// with `with_values()`, `with_named_values()`, or `fit(guess_init=True)`.
    ///
    /// Also accepts the name of a built-in circuit, e.g. `"randles"`; see
    /// `ml_circuits()`.
    #[new]
    fn new(s: &str) -> PyResult<Circuit> {
        let text = models::resolve_alias(s).unwrap_or(s);
        let node = circuit::parse(text)
            .map_err(|e| PyValueError::new_err(circuit::describe_parse_error(text, &e)))?;
        Ok(Circuit {
            node,
            values_supplied: false,
        })
    }

    /// Names accepted by the constructor that have trained initial-parameter
    /// models, and so get a guessed starting point from `fit()` by default.
    #[staticmethod]
    fn ml_circuits() -> Vec<&'static str> {
        models::names()
    }

    /// Machine-learning guess of starting parameters for this circuit's
    /// topology, in `param_names()` order.
    ///
    /// Series and parallel elements may be written with any order and labels:
    /// `(R1,C2)-R3` uses the same model as `R0-(R1,C1)`. `K` and `Zarc` reach
    /// the models trained on the pairs they abbreviate, so `R0-Zarc1` uses the
    /// `R0-(R1,CPE1)` model and gets its guess back as `r`, `tau_k`, `gamma`.
    ///
    /// `weights` loads a `.eisnn` file from disk instead of looking for a
    /// bundled model.
    ///
    /// Raises if no model has been trained for this topology, or if `weights`
    /// was trained for a different one.
    #[pyo3(signature = (frequencies, impedances=None, weights=None))]
    fn guess(
        &self,
        frequencies: &Bound<'_, PyAny>,
        impedances: Option<&Bound<'_, PyAny>>,
        weights: Option<&str>,
    ) -> PyResult<Vec<f64>> {
        let (frequencies, impedances) = spectrum(frequencies, impedances)?;
        guess_params(&self.node, &frequencies, &impedances, weights)
    }

    /// Parameter names, in the same order `with_values()` consumes and
    /// `with_named_values()` expects as keys.
    fn param_names(&self) -> Vec<String> {
        circuit::param_names(&self.node)
    }

    /// Rebuild this circuit with a new flat parameter vector, assigned
    /// positionally in `param_names()` order.
    fn with_values(&self, values: Vec<f64>) -> PyResult<Circuit> {
        let expected = circuit::param_count(&self.node);
        if values.len() != expected {
            return Err(PyValueError::new_err(format!(
                "expected {expected} values, got {}",
                values.len()
            )));
        }
        Ok(Circuit::valued(circuit::with_param_values(
            &self.node, &values,
        )))
    }

    /// Rebuild this circuit with parameter values looked up by name (see
    /// `param_names()`). Every name must be present and no unknown names
    /// may be supplied.
    fn with_named_values(&self, values: HashMap<String, f64>) -> PyResult<Circuit> {
        let names = circuit::param_names(&self.node);

        let unknown: Vec<&str> = values
            .keys()
            .filter(|k| !names.contains(k))
            .map(String::as_str)
            .collect();
        let missing: Vec<&str> = names
            .iter()
            .filter(|n| !values.contains_key(*n))
            .map(String::as_str)
            .collect();

        if !unknown.is_empty() || !missing.is_empty() {
            let units = circuit::param_units(&self.node);
            let bounds = circuit::param_bounds(&self.node);
            return Err(PyValueError::new_err(circuit::describe_param_error(
                &names, &units, &bounds, &unknown, &missing,
            )));
        }

        let positional: Vec<f64> = names.iter().map(|name| values[name]).collect();
        Ok(Circuit::valued(circuit::with_param_values(
            &self.node,
            &positional,
        )))
    }

    fn impedance<'py>(
        &self,
        py: Python<'py>,
        frequencies: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let frequencies = frequencies_of(frequencies)?;
        let node = &self.node;
        let result: Vec<Complex64> = py.allow_threads(|| {
            if frequencies.len() >= PARALLEL_THRESHOLD {
                frequencies
                    .par_iter()
                    .map(|f| circuit::impedance(node, TAU * f))
                    .collect()
            } else {
                frequencies
                    .iter()
                    .map(|f| circuit::impedance(node, TAU * f))
                    .collect()
            }
        });
        Ok(result.into_pyarray(py))
    }

    /// Current parameter values, in `param_names()` order.
    fn param_values(&self) -> Vec<f64> {
        circuit::param_values(&self.node)
    }

    /// Default physical-validity `(lo, hi)` bounds per parameter, in
    /// `param_names()` order. `hi` is `inf` for open-bound parameters.
    fn param_bounds(&self) -> Vec<(f64, f64)> {
        circuit::param_bounds(&self.node)
    }

    /// Physical units per parameter
    fn param_units(&self) -> Vec<&'static str> {
        circuit::param_units(&self.node)
    }

    /// Weighted residual vector (interleaved `[re0, im0, re1, im1, ...]`) for an
    /// arbitrary parameter vector -- the same building block `fit()` uses
    /// internally for Levenberg-Marquardt, exposed so an external optimizer (e.g.
    /// `scipy.optimize.least_squares`) can drive this circuit's math directly.
    #[pyo3(signature = (params, frequencies, impedances=None, weight="modulus"))]
    fn residuals(
        &self,
        py: Python<'_>,
        params: Vec<f64>,
        frequencies: &Bound<'_, PyAny>,
        impedances: Option<&Bound<'_, PyAny>>,
        weight: &str,
    ) -> PyResult<Vec<f64>> {
        let (frequencies, impedances) = spectrum(frequencies, impedances)?;
        let weighting = parse_weighting(weight)?;
        let node = &self.node;
        let omegas: Vec<f64> = frequencies.iter().map(|f| TAU * f).collect();
        let weights = fit::compute_weights(&impedances, weighting);
        Ok(py.allow_threads(|| {
            fit::residuals(
                node,
                &params,
                &omegas,
                &impedances,
                &weights,
                &Default::default(),
            )
        }))
    }

    /// Central-difference Jacobian of `residuals()` at `params`, shape `(2 *
    /// len(frequencies), len(params))` -- rows are residuals, columns are
    /// parameters, matching what `scipy.optimize.least_squares(jac=...)` expects.
    #[pyo3(signature = (params, frequencies, impedances=None, weight="modulus"))]
    fn jacobian(
        &self,
        py: Python<'_>,
        params: Vec<f64>,
        frequencies: &Bound<'_, PyAny>,
        impedances: Option<&Bound<'_, PyAny>>,
        weight: &str,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (frequencies, impedances) = spectrum(frequencies, impedances)?;
        let weighting = parse_weighting(weight)?;
        let node = &self.node;
        let omegas: Vec<f64> = frequencies.iter().map(|f| TAU * f).collect();
        let weights = fit::compute_weights(&impedances, weighting);
        let columns = py.allow_threads(|| {
            fit::jacobian_columns(
                node,
                &params,
                &omegas,
                &impedances,
                &weights,
                &Default::default(),
            )
        });
        let n_rows = columns.first().map_or(0, Vec::len);
        Ok((0..n_rows)
            .map(|i| columns.iter().map(|col| col[i]).collect())
            .collect())
    }

    /// Fit this circuit's parameters to a measured spectrum.
    ///
    /// Accepts two sequences of frequencies and impedances, or a single battery
    /// data format dataframe (`impedances` left as None).
    ///
    /// `guess_init` unset means start from a machine-learning guess whenever the
    /// circuit is not explicitly given starting parameters. If no model has been
    /// trained for the topology it warns and starts from placeholder values.
    ///
    /// `True` always guesses, and raises rather than warns when it cannot.
    /// `False` never guesses.
    #[pyo3(signature = (
        frequencies, impedances=None, guess_init=None, weights=None,
        weight="modulus", method="levenberg_marquardt",
        max_iterations=200, ftol=1e-8, xtol=1e-8,
        num_particles=200, generations=1000,
        nelder_mead_iterations=2000,
        de_evaluations=20_000,
        sa_iterations=5000, sa_initial_temperature=2.0,
        basin_hopping_hops=20, basin_hopping_step_size=1.0, basin_hopping_temperature=1.0,
        seed=None,
    ))]
    #[allow(clippy::too_many_arguments)] // mirrors the Python-facing keyword-argument surface
    fn fit(
        &self,
        py: Python<'_>,
        frequencies: &Bound<'_, PyAny>,
        impedances: Option<&Bound<'_, PyAny>>,
        guess_init: Option<bool>,
        weights: Option<&str>,
        weight: &str,
        method: &str,
        max_iterations: u32,
        ftol: f64,
        xtol: f64,
        num_particles: usize,
        generations: u64,
        nelder_mead_iterations: u64,
        de_evaluations: usize,
        sa_iterations: u64,
        sa_initial_temperature: f64,
        basin_hopping_hops: u32,
        basin_hopping_step_size: f64,
        basin_hopping_temperature: f64,
        seed: Option<u64>,
    ) -> PyResult<FitResult> {
        let (frequencies, impedances) = spectrum(frequencies, impedances)?;
        let weighting = parse_weighting(weight)?;

        // unset means guess whenever there is nothing better to start from
        let wants_guess = guess_init.unwrap_or(!self.values_supplied || weights.is_some());
        let no_model = weights.is_none() && models::find_for_topology(&self.node).is_none();
        let node = if !wants_guess {
            self.node.clone()
        } else if guess_init.is_none() && no_model {
            // only an explicit guess_init=True makes an untrained circuit an error
            warn(py, &models::describe_fallback())?;
            self.node.clone()
        } else {
            let values = guess_params(&self.node, &frequencies, &impedances, weights)?;
            circuit::with_param_values(&self.node, &values)
        };

        let outcome = py
            .allow_threads(|| match method {
                "levenberg_marquardt" => {
                    let options = FitOptions {
                        max_iterations,
                        ftol,
                        xtol,
                        gtol: 1e-8,
                    };
                    fit::levenberg_marquardt_fit(
                        &node,
                        &frequencies,
                        &impedances,
                        weighting,
                        &options,
                    )
                }
                "particle_swarm" => fit::particle_swarm_fit(
                    &node,
                    &frequencies,
                    &impedances,
                    weighting,
                    num_particles,
                    generations,
                    seed,
                ),
                "nelder_mead" => fit::nelder_mead_fit(
                    &node,
                    &frequencies,
                    &impedances,
                    weighting,
                    nelder_mead_iterations,
                ),
                "differential_evolution" => fit::differential_evolution_fit(
                    &node,
                    &frequencies,
                    &impedances,
                    weighting,
                    de_evaluations,
                ),
                "simulated_annealing" => fit::simulated_annealing_fit(
                    &node,
                    &frequencies,
                    &impedances,
                    weighting,
                    sa_iterations,
                    sa_initial_temperature,
                    seed,
                ),
                "basin_hopping" => fit::basin_hopping_fit(
                    &node,
                    &frequencies,
                    &impedances,
                    weighting,
                    basin_hopping_hops,
                    basin_hopping_step_size,
                    basin_hopping_temperature,
                    seed,
                ),
                other => Err(fit::FitError::UnknownMethod(other.to_string())),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let params: HashMap<String, f64> = outcome
            .param_names
            .iter()
            .cloned()
            .zip(outcome.params.iter().copied())
            .collect();
        let stderr: Option<HashMap<String, f64>> = outcome
            .stderr
            .map(|se| outcome.param_names.iter().cloned().zip(se).collect());

        Ok(FitResult {
            circuit: Circuit::valued(outcome.node),
            params,
            stderr,
            success: outcome.success,
            iterations: outcome.iterations,
            impedance_evals: outcome.impedance_evals,
            cost: outcome.cost,
            chi_square: outcome.chi_square,
        })
    }

    fn __repr__(&self) -> String {
        let names = circuit::param_names(&self.node);
        let values = circuit::param_values(&self.node);
        let units = circuit::param_units(&self.node);
        let bounds = circuit::param_bounds(&self.node);
        format!(
            "Circuit ({} parameter{})\n{}",
            names.len(),
            if names.len() == 1 { "" } else { "s" },
            circuit::describe_params(&names, &values, &units, &bounds)
        )
    }
}

#[pyclass(module = "fasteis")]
#[derive(Clone)]
pub struct FitResult {
    #[pyo3(get)]
    pub circuit: Circuit,
    #[pyo3(get)]
    pub params: HashMap<String, f64>,
    #[pyo3(get)]
    pub stderr: Option<HashMap<String, f64>>,
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub iterations: u64,
    #[pyo3(get)]
    pub impedance_evals: u64,
    #[pyo3(get)]
    pub cost: f64,
    #[pyo3(get)]
    pub chi_square: f64,
}

#[pymethods]
impl FitResult {
    fn __repr__(&self) -> String {
        format!(
            "FitResult(success={}, iterations={}, impedance_evals={}, chi_square={:.6e}, params={:?})",
            self.success, self.iterations, self.impedance_evals, self.chi_square, self.params
        )
    }
}

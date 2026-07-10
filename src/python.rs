use std::collections::HashMap;
use std::f64::consts::TAU;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::circuit::{self, Node, Series};
use crate::elements::Element;
use crate::fit::{self, FitOptions, Weighting};

/// Below this many frequency points, rayon's ~20 us overhead outweighs
/// the benefit of parallelizing, so we fall back to a plain sequential map.
/// Depends on circuit complexity, R breaks even at 12_000, randles at 1_500,
/// more complex would be lower.
const PARALLEL_THRESHOLD: usize = 1_000;

#[pyclass]
#[derive(Clone)]
pub struct Circuit {
    pub node: Series,
}

fn leaf(element: Element) -> Series {
    vec![Node::Element(element, None)]
}

#[allow(non_snake_case)] // element codes intentionally mirror impedance.py's naming (R, C, CPE, ...)
#[pymethods]
impl Circuit {
    #[staticmethod]
    fn R(r: f64) -> Circuit {
        Circuit { node: leaf(Element::R { r }) }
    }

    #[staticmethod]
    fn C(c: f64) -> Circuit {
        Circuit { node: leaf(Element::C { c }) }
    }

    #[staticmethod]
    fn L(l: f64) -> Circuit {
        Circuit { node: leaf(Element::L { l }) }
    }

    #[staticmethod]
    fn La(l: f64, alpha: f64) -> Circuit {
        Circuit { node: leaf(Element::La { l, alpha }) }
    }

    #[staticmethod]
    fn CPE(q: f64, alpha: f64) -> Circuit {
        Circuit { node: leaf(Element::Cpe { q, alpha }) }
    }

    #[staticmethod]
    fn W(aw: f64) -> Circuit {
        Circuit { node: leaf(Element::W { aw }) }
    }

    #[staticmethod]
    fn Wo(z0: f64, tau: f64) -> Circuit {
        Circuit { node: leaf(Element::Wo { z0, tau }) }
    }

    #[staticmethod]
    fn Ws(z0: f64, tau: f64) -> Circuit {
        Circuit { node: leaf(Element::Ws { z0, tau }) }
    }

    #[staticmethod]
    fn G(rg: f64, tg: f64) -> Circuit {
        Circuit { node: leaf(Element::G { rg, tg }) }
    }

    #[staticmethod]
    fn Gs(rg: f64, tg: f64, phi: f64) -> Circuit {
        Circuit { node: leaf(Element::Gs { rg, tg, phi }) }
    }

    #[staticmethod]
    fn K(r: f64, tau_k: f64) -> Circuit {
        Circuit { node: leaf(Element::K { r, tau_k }) }
    }

    #[staticmethod]
    fn Zarc(r: f64, tau_k: f64, gamma: f64) -> Circuit {
        Circuit { node: leaf(Element::Zarc { r, tau_k, gamma }) }
    }

    #[staticmethod]
    fn TLMQ(r_ion: f64, qs: f64, gamma: f64) -> Circuit {
        Circuit { node: leaf(Element::Tlmq { r_ion, qs, gamma }) }
    }

    #[staticmethod]
    fn T(a_coeff: f64, b_coeff: f64, a_param: f64, b_param: f64) -> Circuit {
        Circuit { node: leaf(Element::T { a_coeff, b_coeff, a_param, b_param }) }
    }

    #[staticmethod]
    fn series(elements: Vec<Circuit>) -> Circuit {
        Circuit { node: elements.into_iter().flat_map(|c| c.node).collect() }
    }

    #[staticmethod]
    fn parallel(elements: Vec<Circuit>) -> Circuit {
        Circuit { node: vec![Node::Parallel(elements.into_iter().map(|c| c.node).collect())] }
    }

    /// Parse a circuit topology string, e.g. `"R0-p(R1,Cpe1)"` or
    /// `"R0-p(R1-C1,R2-Cpe2)"`. The string carries no parameter values --
    /// every element gets a placeholder default; set real values afterward
    /// with `with_values()` or `with_named_values()`.
    #[staticmethod]
    fn from_string(s: &str) -> PyResult<Circuit> {
        let node = circuit::parse(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Circuit { node })
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
        Ok(Circuit { node: circuit::with_param_values(&self.node, &values) })
    }

    /// Rebuild this circuit with parameter values looked up by name (see
    /// `param_names()`). Every name must be present and no unknown names
    /// may be supplied.
    fn with_named_values(&self, values: HashMap<String, f64>) -> PyResult<Circuit> {
        let names = circuit::param_names(&self.node);

        let unknown: Vec<&String> = values.keys().filter(|k| !names.contains(k)).collect();
        if !unknown.is_empty() {
            return Err(PyValueError::new_err(format!("unknown parameter name(s): {unknown:?}")));
        }

        let mut positional = Vec::with_capacity(names.len());
        for name in &names {
            match values.get(name) {
                Some(&v) => positional.push(v),
                None => return Err(PyValueError::new_err(format!("missing value for parameter {name:?}"))),
            }
        }
        Ok(Circuit { node: circuit::with_param_values(&self.node, &positional) })
    }

    fn impedance<'py>(&self, py: Python<'py>, frequencies: Vec<f64>) -> Bound<'py, PyArray1<Complex64>> {
        let node = &self.node;
        let result: Vec<Complex64> = py.allow_threads(|| {
            if frequencies.len() >= PARALLEL_THRESHOLD {
                frequencies.par_iter().map(|f| circuit::impedance(node, TAU * f)).collect()
            } else {
                frequencies.iter().map(|f| circuit::impedance(node, TAU * f)).collect()
            }
        });
        result.into_pyarray(py)
    }

    #[pyo3(signature = (frequencies, impedances, weight="modulus", max_iterations=200, ftol=1e-10, xtol=1e-10))]
    #[allow(clippy::too_many_arguments)] // mirrors the Python-facing keyword-argument surface
    fn fit(
        &self,
        py: Python<'_>,
        frequencies: Vec<f64>,
        impedances: Vec<Complex64>,
        weight: &str,
        max_iterations: u32,
        ftol: f64,
        xtol: f64,
    ) -> PyResult<FitResult> {
        let weighting = match weight {
            "modulus" => Weighting::Modulus,
            "unit" => Weighting::Unit,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown weight scheme {other:?}, expected \"modulus\" or \"unit\""
                )));
            }
        };
        let options = FitOptions { max_iterations, ftol, xtol, gtol: 1e-10 };
        let node = self.node.clone();
        let outcome = py
            .allow_threads(|| fit::levenberg_marquardt_fit(&node, &frequencies, &impedances, weighting, &options))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let params: HashMap<String, f64> =
            outcome.param_names.iter().cloned().zip(outcome.params.iter().copied()).collect();
        let stderr: Option<HashMap<String, f64>> =
            outcome.stderr.map(|se| outcome.param_names.iter().cloned().zip(se).collect());

        Ok(FitResult {
            circuit: Circuit { node: outcome.node },
            params,
            stderr,
            success: outcome.success,
            iterations: outcome.iterations,
            cost: outcome.cost,
            chi_square: outcome.chi_square,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.node)
    }
}

#[pyclass]
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
    pub cost: f64,
    #[pyo3(get)]
    pub chi_square: f64,
}

#[pymethods]
impl FitResult {
    fn __repr__(&self) -> String {
        format!(
            "FitResult(success={}, iterations={}, chi_square={:.6e}, params={:?})",
            self.success, self.iterations, self.chi_square, self.params
        )
    }
}

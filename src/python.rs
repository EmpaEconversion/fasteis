use std::collections::HashMap;
use std::f64::consts::TAU;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::circuit::Node;
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
    pub node: Node,
}

#[allow(non_snake_case)] // element codes intentionally mirror impedance.py's naming (R, C, CPE, ...)
#[pymethods]
impl Circuit {
    #[staticmethod]
    fn R(r: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::R { r }) }
    }

    #[staticmethod]
    fn C(c: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::C { c }) }
    }

    #[staticmethod]
    fn L(l: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::L { l }) }
    }

    #[staticmethod]
    fn La(l: f64, alpha: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::La { l, alpha }) }
    }

    #[staticmethod]
    fn CPE(q: f64, alpha: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::Cpe { q, alpha }) }
    }

    #[staticmethod]
    fn W(aw: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::W { aw }) }
    }

    #[staticmethod]
    fn Wo(z0: f64, tau: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::Wo { z0, tau }) }
    }

    #[staticmethod]
    fn Ws(z0: f64, tau: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::Ws { z0, tau }) }
    }

    #[staticmethod]
    fn G(rg: f64, tg: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::G { rg, tg }) }
    }

    #[staticmethod]
    fn Gs(rg: f64, tg: f64, phi: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::Gs { rg, tg, phi }) }
    }

    #[staticmethod]
    fn K(r: f64, tau_k: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::K { r, tau_k }) }
    }

    #[staticmethod]
    fn Zarc(r: f64, tau_k: f64, gamma: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::Zarc { r, tau_k, gamma }) }
    }

    #[staticmethod]
    fn TLMQ(r_ion: f64, qs: f64, gamma: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::Tlmq { r_ion, qs, gamma }) }
    }

    #[staticmethod]
    fn T(a_coeff: f64, b_coeff: f64, a_param: f64, b_param: f64) -> Circuit {
        Circuit { node: Node::Leaf(Element::T { a_coeff, b_coeff, a_param, b_param }) }
    }

    #[staticmethod]
    fn series(elements: Vec<Circuit>) -> Circuit {
        Circuit { node: Node::Series(elements.into_iter().map(|c| c.node).collect()) }
    }

    #[staticmethod]
    fn parallel(elements: Vec<Circuit>) -> Circuit {
        Circuit { node: Node::Parallel(elements.into_iter().map(|c| c.node).collect()) }
    }

    fn impedance<'py>(&self, py: Python<'py>, frequencies: Vec<f64>) -> Bound<'py, PyArray1<Complex64>> {
        let node = &self.node;
        let result: Vec<Complex64> = py.allow_threads(|| {
            if frequencies.len() >= PARALLEL_THRESHOLD {
                frequencies.par_iter().map(|f| node.impedance(TAU * f)).collect()
            } else {
                frequencies.iter().map(|f| node.impedance(TAU * f)).collect()
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

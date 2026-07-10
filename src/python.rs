use std::f64::consts::TAU;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::circuit::Node;
use crate::elements::Element;

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

    fn __repr__(&self) -> String {
        format!("{:?}", self.node)
    }
}

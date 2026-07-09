use std::f64::consts::TAU;

use num_complex::Complex64;
use pyo3::prelude::*;

use crate::circuit::Node;
use crate::elements::Element;

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

    fn impedance(&self, frequencies: Vec<f64>) -> Vec<Complex64> {
        frequencies.iter().map(|f| self.node.impedance(TAU * f)).collect()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.node)
    }
}

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod circuit;
mod elements;
mod fit;
mod models;
mod nn;
mod python;

use elements::Element;
use python::{parallel, series, Circuit, FitResult};

/// Element variants, re-exported to allow e.g. `from fasteis import R, C`
const ELEMENT_NAMES: [&str; 14] = [
    "R", "C", "L", "La", "Cpe", "W", "Wo", "Ws", "G", "Gs", "K", "Zarc", "Tlmq", "T",
];

#[pymodule]
fn fasteis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Element>()?;
    m.add_class::<Circuit>()?;
    m.add_class::<FitResult>()?;
    m.add_function(wrap_pyfunction!(series, m)?)?;
    m.add_function(wrap_pyfunction!(parallel, m)?)?;

    let element = m.py().get_type::<Element>();
    for name in ELEMENT_NAMES {
        m.add(name, element.getattr(name)?)?;
    }
    Ok(())
}

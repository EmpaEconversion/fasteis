use pyo3::prelude::*;

mod circuit;
mod elements;
mod fit;
mod models;
mod nn;
mod python;

use python::{Circuit, FitResult};

#[pymodule]
fn fasteis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Circuit>()?;
    m.add_class::<FitResult>()?;
    Ok(())
}

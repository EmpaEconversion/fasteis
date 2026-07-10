use pyo3::prelude::*;

mod circuit;
mod elements;
mod fit;
mod python;

use python::{Circuit, FitResult};

#[pymodule]
fn eis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Circuit>()?;
    m.add_class::<FitResult>()?;
    Ok(())
}

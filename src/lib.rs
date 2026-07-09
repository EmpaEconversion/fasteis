use pyo3::prelude::*;

mod circuit;
mod elements;
mod python;

use python::Circuit;

#[pymodule]
fn eis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Circuit>()?;
    Ok(())
}

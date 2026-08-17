# Fitting a circuit

Use `Circuit.fit()`

```python
from fasteis import Circuit

circuit = Circuit("R1-(R2,C2)")

res = circuit.fit(f, Z)
```
where `f` and `Z` are sequences of frequencies and complex impedances.

You can also pass a 'Battery Data Format' style dataframe directly:

```python
from fasteis import Circuit
import bdf

circuit = Circuit("R1-(R2,C2)")

df = bdf.read("my/bdf/file.parquet")

res = circuit.fit(df)
```

## Machine learning guesses

`fasteis` has small convolutional neural networks trained on specific common circuits.
It can use these models to guess good initial parameters to the fit,
meaning faster, more robust fits, and no need to manually adjust input parameters.

This happens by default if initial parameters are not supplied to the circuit (as above).

To see the available models, use `Circuit.ml_circuits()`.

A model will be used if it matches your circuit topology, even if elements are
placed in a different way.

E.g. the `"rc"` circuit `"R0-(R1,C1)"` will match the following:

* `"R1-(R3,C7)"` - different labels
* `"(C0,R1)-R2"` - different order
* `"R1-K1"` - equivalent elements used

You can force or disable the machine learning guess with:
```python
circuit.fit(f, Z, guess_init=False)
```

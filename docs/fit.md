# Fitting a circuit

Use `Circuit.fit()`

```python
from fasteis import Circuit

circ = Circuit("R1-(R2,C2)")

res = circ.fit(f, Z)
```
where `f` and `Z` are sequences of frequencies and complex impedances.

You can also pass a 'Battery Data Format' style dataframe directly:

```python
from fasteis import Circuit
import bdf

circ = Circuit("R1-(R2,C2)")

df = bdf.read("my/bdf/file.parquet")

res = circ.fit(df)
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
circ.fit(f, Z, guess_init=False)
```

## Fixing parameters

Hold parameters constant during the fit with `fixed`, using the names from
`Circuit.param_names()`.

A dict holds each parameter at the given value:

```python
circ = Circuit("sei_randles")

res = circ.fit(f, Z, fixed={"R0.r": 100.0})
```

Parameters start from the machine learning guess with `R0.r` replaced by 100.
The guess does not change when fixing parameters.

You can also pass a list of parameter names for circuits with existing values:

```python
circ = Circuit("R0-(R1,C1)").with_values({"R0.r": 10.0, "R1.r": 20.0, "C1.c": 1.0})
res = circ.fit(f, Z, fixed=["R0.r"])
```

Fixed parameters appear in `res.params` with their held values, and are left
out of `res.stderr`.

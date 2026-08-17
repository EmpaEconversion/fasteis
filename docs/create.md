# Creating a circuit

## Define from string

The most convenient method is to define a circuit with a string

```python
from fasteis import Circuit

circuit = Circuit("R1-(R2,C2)")
```

Here:

* `R1` means a resistor `R` with label `1`
* `R2` means a resistor `R` with label `2`
* `C2` means a capacitor `C` with label `2`
* A hyphen `x-y` means element `x` and `y` are in series
* Brackets `(x,y)` mean element `x` and `y` are in parallel

Labels can be any number, as long as elements do not have the same type and label.

Circuits can be arbitrarily nested and complex.
See [`Element`](api.md#fasteis.Element) in the API reference for the available elements.

E.g. for a suppressed two arc circuit with finite length Warburg diffusion:

```python
from fasteis import Circuit

circuit = Circuit("R0-(R1,CPE1)-(R2-Wo2,CPE2)")
```

## Setting parameter values

Use `with_values(...)` and supply a list of values in order of elements and their attributes:

```python
from fasteis import Circuit

circuit = Circuit("R1-(R2,C2)").with_values([10.0, 20.0, 1e-3])
```

Or use `with_named_values(...)` and supply a dict of `"{element}.{attribute}": value`:


```python
circuit = Circuit("R1-(R2,C2)").with_named_values(
    {
        "R1.r": 10.0,
        "R2.r": 20.0,
        "C2.c": 1e-3,
    }
)
```

## Define from Python objects
You can also create a circuit from element objects directly by giving `Series`
and `Parallel` lists of `Element` objects.

The `Element` objects accept positional or named arguments, e.g. `R(10.0)` and
`R(r=10.0)` are both valid.

```python
from fasteis import Series, Parallel, R, C

circuit = Series([R(10.0), Parallel([R(5.0),C(1e-3)])])
```


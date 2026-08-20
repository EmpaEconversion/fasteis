<h1 align="center">
  <img src="https://github.com/user-attachments/assets/87f057ee-d9a8-41fc-bdc2-81f848123185" width="400" align="center" alt="fasteis">
</h1>

<br>

> [!WARNING]
> In early development, not yet suitable for use in production

A library for simulating EIS with equivalent circuits.

`fasteis` uses a few tricks to make EIS fitting faster and more reliable:

- Small convolutional neural networks are used to automatically get good initial fit parameters
- It is written in Rust with fast and numerically stable maths

Builds on the excellent work from
[`impedance.py`](https://github.com/ECSHackWeek/impedance.py) and
[`PyEIS`](https://github.com/kbknudsen/PyEIS).

Quickstart:
```bash
pip install fasteis
```
```python
import fasteis

circuit = fasteis.Circuit("R0-(CPE1,R1-W1)")
result = circuit.fit(f, Z)
```

# fasteis

A Python library for simulating EIS with equivalent circuits.

`fasteis` uses a few tricks to make EIS fitting faster and more reliable:

- Small convolutional neural networks are used to automatically get good initial fit parameters.
- It is written in Rust with fast and numerically stable maths.

Builds on the excellent work from
[`impedance.py`](https://github.com/ECSHackWeek/impedance.py) and
[`PyEIS`](https://github.com/kbknudsen/PyEIS).

The project is at an early stage, and the API may change at any time without a
major version bump.

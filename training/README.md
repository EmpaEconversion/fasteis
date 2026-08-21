# ML parameter guessing

Scripts to train a neural network which guesses circuit parameters from
measured EIS data, used as an initial guess for `Circuit.fit()`.

See the [Training](https://empaeconversion.github.io/fasteis/training/) and
[Models](https://empaeconversion.github.io/fasteis/models/) sections of the
docs for details, and `Adding a circuit` in the Training page for the workflow
to add a new one.

Training needs torch (`uv sync --all-extras`).

# Copyright © 2026, Empa.
"""Create schematic diagrams and Nyquist plots for trained circuits.

Writes into `docs/assets/circuits/`. The schematic parses
`TrainingCircuit.circuit_str` and draws the circuit with `schemdraw`.

Saves light and dark mode diagrams and plots.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import schemdraw
import schemdraw.elements as elm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fasteis
from training import circuits

ELEMENT_SYMBOLS = {
    "R": elm.Resistor,
    "C": elm.Capacitor,
    "L": elm.Inductor,
    "CPE": elm.CPE,
}

UNIT = 1.5
BRANCH_SPACING = 1.25
LEAD_LENGTH = 0.5

_TOKEN_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split on `sep`, ignoring occurrences nested inside parentheses."""
    parts, depth, current = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == sep and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    return parts


def _parse_term(term: str) -> tuple:
    term = term.strip()
    if term.startswith("(") and term.endswith(")"):
        branches = _split_top_level(term[1:-1], ",")
        return ("parallel", [_parse(b) for b in branches])
    m = _TOKEN_RE.match(term)
    if not m:
        raise ValueError(f"cannot parse circuit element {term!r}")
    return ("element", m.group(1), m.group(2))


def _parse(circuit_str: str) -> tuple:
    """Parse a circuit topology string into a tree of nested tuples.

    Each node is `("element", type, label)`, `("series", children)` or `("parallel", children)`.
    """
    terms = [_parse_term(t) for t in _split_top_level(circuit_str, "-")]
    return terms[0] if len(terms) == 1 else ("series", terms)


def _width(node: tuple) -> float:
    """Horizontal span of a node, in element-widths."""
    kind = node[0]
    if kind == "element":
        return 1
    if kind == "series":
        return sum(_width(c) for c in node[1])
    return max(_width(c) for c in node[1])  # parallel


def _draw(d: schemdraw.Drawing, node: tuple, x: float, y: float) -> float:
    """Draw `node` starting at `(x, y)`, growing right. Returns the end x."""
    kind = node[0]
    if kind == "element":
        etype, label = node[1], node[2]
        symbol = ELEMENT_SYMBOLS.get(etype, elm.RBox)
        d += symbol().at((x, y)).right().length(UNIT).label(f"{etype}{label}")
        return x + UNIT

    if kind == "series":
        cx = x
        for child in node[1]:
            cx = _draw(d, child, cx, y)
        return cx

    # parallel: even up the branches on two vertical rails, one node each side
    branches = node[1]
    total_w = _width(node) * UNIT
    end_x = x + total_w
    n = len(branches)
    offsets = [(i - (n - 1) / 2) * BRANCH_SPACING for i in range(n)]

    d += elm.Dot().at((x, y))
    d += elm.Dot().at((end_x, y))
    for branch, dy in zip(branches, offsets):
        by = y + dy
        if dy != 0:
            d += elm.Line().at((x, y)).to((x, by))

        pad = (total_w - _width(branch) * UNIT) / 2
        bx = x
        if pad > 0:
            d += elm.Line().at((x, by)).right().length(pad)
            bx += pad
        end_bx = _draw(d, branch, bx, by)
        if pad > 0:
            d += elm.Line().at((end_bx, by)).right().length(pad)
            end_bx += pad

        if dy != 0:
            d += elm.Line().at((end_bx, by)).to((end_bx, y))
    return end_x


def render_schematic(circuit_str: str, color: str, out_path: Path) -> None:
    """Render a circuit topology string to a transparent SVG file."""
    tree = _parse(circuit_str)
    d = schemdraw.Drawing()
    d.config(color=color, fontsize=12)

    x = y = 0.0
    d += elm.Dot().at((x, y))
    d += elm.Line().at((x, y)).right().length(LEAD_LENGTH)
    x += LEAD_LENGTH
    end_x = _draw(d, tree, x, y)
    d += elm.Line().at((end_x, y)).right().length(LEAD_LENGTH)
    d += elm.Dot().at((end_x + LEAD_LENGTH, y))

    d.save(str(out_path), transparent=True)
    plt.close("all")


# One value per parameter name, shared by every circuit, so e.g. the first
# arc of `two_rq` and `randles` matches `rq` exactly.
_TAU1, _TAU2, _ALPHA = 1e-4, 1e-1, 0.7
_R1, _R2 = 1.0, 1.5
NYQUIST_PARAMS: dict[str, float] = {
    "R0.r": 0.3,
    "L0.l": 5e-8,
    "R1.r": _R1,
    "C1.c": _TAU1 / _R1,
    "CPE1.q": _TAU1**_ALPHA / _R1,
    "CPE1.alpha": _ALPHA,
    "W1.aw": 0.2,
    "R2.r": _R2,
    "C2.c": _TAU2 / _R2,
    "CPE2.q": _TAU2**_ALPHA / _R2,
    "CPE2.alpha": _ALPHA,
    "W2.aw": 0.2,
    "Wo2.z0": 0.8,
    "Wo2.tau": 10.0,
}
NYQUIST_FREQS = np.logspace(-2, 6, 600)

# Shared limits so every plot has the same scale and size; circuits with an
# inductor shift the window down to show the tail below the real axis.
X_LIM = (-0.1, 3.7)
Y_SPAN = 1.45
Y_MIN, Y_MIN_INDUCTIVE = -0.1, -0.45
FIG_WIDTH = 3.4


def render_nyquist(circuit_str: str, color: str, axis_color: str, out_path: Path) -> None:
    """Render a noise-free Nyquist curve with light axes to a transparent SVG file."""
    circuit = fasteis.Circuit(circuit_str)
    circuit = circuit.with_named_values({k: NYQUIST_PARAMS[k] for k in circuit.param_names()})
    z = np.asarray(circuit.impedance(list(NYQUIST_FREQS)))

    y_min = Y_MIN_INDUCTIVE if (-z.imag).min() < -0.05 else Y_MIN
    y_lim = (y_min, y_min + Y_SPAN)
    x_span = X_LIM[1] - X_LIM[0]
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * Y_SPAN / x_span))
    ax = fig.add_axes((0, 0, 1, 1))

    arrow = {"arrowstyle": "-|>", "color": axis_color, "lw": 1, "mutation_scale": 8}
    ax.annotate("", xy=(X_LIM[1], 0), xytext=(0, 0), arrowprops=arrow)
    ax.annotate("", xy=(0, y_lim[1]), xytext=(0, y_lim[0]), arrowprops=arrow)
    ax.text(X_LIM[1], 0.04, "Re(Z)", color=axis_color, fontsize=8, ha="right", va="bottom")
    ax.text(0.06, y_lim[1], "−Im(Z)", color=axis_color, fontsize=8, ha="left", va="top")

    ax.plot(z.real, -z.imag, color=color, linewidth=2, solid_capstyle="round")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


def main() -> None:
    """Render every trained circuit's diagrams into `docs/assets/circuits/`."""
    out_dir = Path("docs/assets/circuits")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, circuit in circuits.CIRCUITS.items():
        s = circuit.circuit_str
        render_schematic(s, "#1a1a1a", out_dir / f"{name}.svg")
        render_schematic(s, "#e6e6e6", out_dir / f"{name}-dark.svg")
        render_nyquist(s, "#1a1a1a", "#b0b0b0", out_dir / f"{name}-nyquist.svg")
        render_nyquist(s, "#e6e6e6", "#6b6b6b", out_dir / f"{name}-nyquist-dark.svg")
        print(f"{name:<15} {s}")


if __name__ == "__main__":
    main()

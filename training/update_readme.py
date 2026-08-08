"""Renders benchmark results into README.md.

`benchmark.py` writes one json per circuit to `training/results/`. This turns
them into markdown and substitutes it between the markers in the README.

    <!-- results:randles -->
    ...replaced...
    <!-- /results:randles -->
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import circuits

FITTERS = (
    ("plain_lm", "Plain LM:"),
    ("circuit_fit", "`Circuit.fit()` / smart LM, which screens candidate starts:"),
)


def _table(rows: list[dict]) -> list[str]:
    """Get the markdown results table as list of strings."""
    out = [
        "| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        cells = [
            r["name"],
            f"{100 * r['converged']:.2f}%",
            f"{r['median_excess']:.0f}",
            f"{r['p90_excess']:.0f}",
            f"{r['p99_excess']:.0f}",
            f"{r['median_evaluations']:.0f}",
        ]
        if "ml" in r["name"]:
            cells = [f"**{c}**" for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    return out


def render(results: dict) -> str:
    """Markdown for one circuit's results."""
    lines = [
        f"`{results['circuit_str']}`, {results['n_spectra']} synthetic spectra. "
        f"Inference costs {results['inference_ms']:.2f} ms/spectrum against "
        f"{results['floor_fit_ms']:.2f} ms for the fit it starts.",
        "",
    ]
    for key, heading in FITTERS:
        rows = results["fitters"].get(key)
        if not rows:
            continue
        lines += [heading, *_table(rows), ""]

    error = results["param_error_pct"]
    lines += [
        "Relative error of the ml guess, before fitting (%):",
        "| | " + " | ".join(f"`{name}`" for name in error) + " |",
        "|---" * (len(error) + 1) + "|",
    ]
    for stat in ("median", "p90", "p99"):
        cells = " | ".join(f"{error[name][stat]:.1f}" for name in error)
        lines.append(f"| {stat} | {cells} |")

    lines += ["", f"<sub>generated {results['generated']} by benchmark.py</sub>"]
    return "\n".join(lines)


def substitute(text: str, name: str, body: str) -> tuple[str, bool]:
    """Replace the marked block for one circuit. Returns (text, found)."""
    pattern = re.compile(
        rf"(<!-- results:{re.escape(name)} -->\n).*?(\n<!-- /results:{re.escape(name)} -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        return text, False
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text), True


def main() -> None:
    """Render every available result file into the README."""
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("training/results"))
    p.add_argument("--readme", type=Path, default=Path("training/README.md"))
    args = p.parse_args()

    text = args.readme.read_text(encoding="utf-8")
    for name in circuits.CIRCUITS:
        path = args.results / f"{name}.json"
        if not path.exists():
            print(f"{name:<10} no results yet, run benchmark.py --circuit {name}")
            continue
        results = json.loads(path.read_text(encoding="utf-8"))
        text, found = substitute(text, name, render(results))
        if found:
            print(f"{name:<10} updated from {path}")
        else:
            print(f"{name:<10} no <!-- results:{name} --> markers in {args.readme}")

    args.readme.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

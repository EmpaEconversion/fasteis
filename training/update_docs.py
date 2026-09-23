# Copyright © 2026, Empa.
"""Renders benchmark results into the Zensical docs.

`benchmark.py` writes one json per circuit to `training/results/`. This turns
them into markdown and substitutes it between the markers in `docs/models/`.

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

from training import circuits, evaluate, priors, serialize_weights

FITTERS = (
    ("plain_lm", "### Plain LM"),
    ("circuit_fit", "### `Circuit.fit()`"),
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


def render_model(results: dict, weights: Path) -> str:
    """One line on the size and cost of a circuit's bundled model."""
    _, tensors = serialize_weights.read(weights)
    n_weights = sum(t.size for name, t in tensors.items() if name.startswith("w."))
    return (
        f"ML model: {n_weights / 1000:.0f}k parameter 1D CNN, trained on synthetic data, "
        f"{results['inference_ms']:.1f} ms per guess. See [Training](../training.md)."
    )


def render(results: dict) -> str:
    """Markdown for one circuit's results."""
    lines = []
    for key, heading in FITTERS:
        rows = results["fitters"].get(key)
        if not rows:
            continue
        lines += [heading, "", *_table(rows), ""]

    error = results["param_error_pct"]
    lines += [
        "### Error of the guess",
        "",
        "Relative error of each guessed parameter before fitting, in %.",
        "",
        "| | " + " | ".join(f"`{name}`" for name in error) + " |",
        "|---" * (len(error) + 1) + "|",
    ]
    for stat in ("median", "p90", "p99"):
        cells = " | ".join(f"{error[name][stat]:.1f}" for name in error)
        lines.append(f"| {stat} | {cells} |")

    return "\n".join(lines)


def render_method(results: dict) -> str:
    """Markdown describing how one circuit's benchmark was run."""
    cfg = priors.DEFAULT
    noise_lo, noise_hi = (100 * 10.0**x for x in cfg.log_noise)
    tol_pct = 100 * (evaluate.CONVERGENCE_TOL - 1)
    alpha = (
        f", CPE exponents ±{evaluate.PERTURB_ALPHA:g}"
        if circuits.get(results["circuit"]).linear_params
        else ""
    )
    factor = f"{evaluate.PERTURB_FACTOR:g}"
    return "\n".join(
        [
            f"Synthetic benchmarks use {results['n_spectra']} spectra drawn from the same "
            "distribution as the training data, with a different seed: "
            f"{cfg.decades[0]:g} to {cfg.decades[1]:g} decade sweeps of {cfg.n_points[0]} to "
            f"{cfg.n_points[1]} points, with {noise_lo:.1f}% to {noise_hi:.0f}% noise.",
            "",
            "- **floor (truth)**: start from the true parameters.",
            "- **library defaults**: start from the `Circuit()` placeholder values.",
            f"- **truth x/div {factor}**: true magnitudes multiplied or divided by {factor} "
            f"at random{alpha}.",
            "- **ml guess**: start from `Circuit.guess()`.",
            "",
            "Plain LM is a single-start Levenberg-Marquardt. `Circuit.fit()` is "
            "LM that also screens candidate starts and restarts on bad fits.",
            "",
            f"**Converged**: final cost within {tol_pct:g}% of the fit from the truth. "
            "**Sweeps**: impedance evaluations of the whole spectrum, including Jacobians. "
            "**Excess**: sweeps beyond the fit from the truth, for converged fits only.",
        ]
    )


def render_library(every: dict[str, dict]) -> str:
    """Summary table for trained circuits."""
    lines = [
        "| name | circuit | params | truth x/div 5 | ml guess | floor | ml excess med "
        "| ml excess p90 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, results in every.items():
        rows = {r["name"]: r for r in results["fitters"]["plain_lm"]}
        floor, defaults, ml = (
            rows["floor (truth)"],
            rows["truth x/div 5"],
            rows["ml guess"],
        )
        lines.append(
            f"| [`{name}`]({name}.md) | `{results['circuit_str']}` "
            f"| {len(results['param_error_pct'])} "
            f"| {100 * defaults['converged']:.1f}% "
            f"| **{100 * ml['converged']:.1f}%** "
            f"| {floor['median_evaluations']:.0f}"
            f"| **{ml['median_excess']:.0f}** | {ml['p90_excess']:.0f} |"
        )
    return "\n".join(lines)


def render_real_data_test(results: dict) -> str:
    """Markdown for one circuit's measured-data results."""
    lines = [
        f"Fitted to {results['n_spectra']} measured spectra. Ground truth is not known, "
        "so 'converged' means within tolerance of the best chi-square reached.",
        "",
        "| source of initial parameters | converged | med sweeps | med ms | med chi2 |",
        "|---|---|---|---|---|",
    ]
    for s in results["strategies"]:
        cells = [
            s["name"],
            f"{100 * s['converged']:.2f}%",
            f"{s['median_sweeps']:.0f}",
            f"{s['median_ms']:.2f}",
            f"{s['median_chi_square']:.3e}",
        ]
        if "ml" in s["name"]:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")

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
    """Render every available result file into the docs."""
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("training/results"))
    p.add_argument("--docs", type=Path, default=Path("docs/models"))
    p.add_argument("--models", type=Path, default=Path("src/models"))
    args = p.parse_args()

    every = {}
    for name in circuits.CIRCUITS:
        path = args.results / f"{name}.json"
        if not path.exists():
            print(f"{name:<15} no results yet, run benchmark.py --circuit {name}")
            continue
        every[name] = json.loads(path.read_text(encoding="utf-8"))

        page = args.docs / f"{name}.md"
        page_text = page.read_text(encoding="utf-8")
        page_text, found = substitute(page_text, name, render(every[name]))
        page_text, _ = substitute(page_text, f"{name}_method", render_method(every[name]))
        model = render_model(every[name], args.models / f"{name}.eisnn")
        page_text, _ = substitute(page_text, f"{name}_model", model)
        print(
            f"{name:<15} updated from {path}"
            if found
            else f"{name:<15} no <!-- results:{name} --> markers in {page}"
        )

        real = args.results / f"{name}_real.json"
        if real.exists():
            body = render_real_data_test(json.loads(real.read_text(encoding="utf-8")))
            page_text, found = substitute(page_text, f"{name}_real", body)
            if found:
                print(f"{name + '_real':<15} updated from {real}")
        page.write_text(page_text, encoding="utf-8")

    if every:
        index = args.docs / "index.md"
        index_text = index.read_text(encoding="utf-8")
        index_text, found = substitute(index_text, "library", render_library(every))
        print(f"library         {'updated' if found else 'no <!-- results:library --> markers'}")
        index.write_text(index_text, encoding="utf-8")


if __name__ == "__main__":
    main()

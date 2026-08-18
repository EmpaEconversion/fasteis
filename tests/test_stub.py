"""Tests that the type stub declares the same public API as the module."""

from __future__ import annotations

import ast
from pathlib import Path

import fasteis

# The source stub (not installed copy, which only refreshes on cargo build)
STUB = Path(__file__).parent.parent / "fasteis.pyi"


def _stub_tree() -> ast.Module:
    return ast.parse(STUB.read_text(encoding="utf-8"))


def _stub_all() -> list[str]:
    """Names listed in the stub's `__all__`."""
    for node in _stub_tree().body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            return [ast.literal_eval(element) for element in node.value.elts]
    raise AssertionError(f"{STUB} has no __all__")


def _stub_top_level_names() -> set[str]:
    """Every name the stub declares at module level."""
    names: set[str] = set()
    for node in _stub_tree().body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_stub_all_matches_runtime() -> None:
    """Stub `__all__` matches runtime `__all__`."""
    assert sorted(_stub_all()) == sorted(fasteis.__all__)


def test_stub_declares_every_export() -> None:
    """Catches an element added to the module but not aliased in the stub."""
    missing = sorted(set(fasteis.__all__) - _stub_top_level_names())
    assert not missing


def test_stub_is_packaged() -> None:
    """The stub and marker must ship for the package to be typed."""
    installed = Path(fasteis.__file__).parent
    assert (installed / "__init__.pyi").is_file()
    assert (installed / "py.typed").is_file()

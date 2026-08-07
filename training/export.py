"""Converts a torch checkpoint into the `.eisnn` container embedded by src/models.rs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import circuits, features, model, scales, serialize_weights


def export(checkpoint: Path, out: Path, dtype: str = "f16") -> None:
    """Write weights, target statistics and scaling rules to `out`."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    circuit = circuits.get(payload["circuit"])
    config = model.Config(**payload["config"])
    net = model.GuessNet(circuit.n_params, config)
    net.load_state_dict(payload["state_dict"])
    net.eval()

    tensors = {
        f"w.{name}": tensor.detach().cpu().numpy().astype(np.float64)
        for name, tensor in net.state_dict().items()
    }
    # unprefixed names are not weights, so serialize_weights keeps them f32
    tensors["target_mean"] = np.asarray(payload["target_mean"], dtype=np.float64)
    tensors["target_std"] = np.asarray(payload["target_std"], dtype=np.float64)
    tensors["scaling"] = circuit.scaling
    tensors["log_params"] = np.array(
        [1.0 if i in circuit.log_params else 0.0 for i in range(circuit.n_params)]
    )

    metadata = {
        "circuit": circuit.circuit_str,
        "param_names": ",".join(circuit.param_names),
        "estimator": str(payload.get("estimator", scales.DEFAULT)),
        "n_grid": str(features.N_GRID),
        "channels": str(config.channels),
        "kernel": str(model.KERNEL),
        "groups": str(config.groups),
        "dilations": ",".join(str(d) for d in config.dilations()),
        "alpha_min": str(circuit.alpha_range[0]),
        "alpha_max": str(circuit.alpha_range[1]),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    serialize_weights.write(out, metadata, tensors, dtype=dtype)

    print(
        f"exported step {payload['step']} -> {out} "
        f"({out.stat().st_size / 1024:.0f} KiB, {dtype}, "
        f"{model.parameter_count(net):,} parameters)"
    )
    print(
        f"  checkpoint scored converged={payload['converged']:.3f} "
        f"median_excess={payload['median_excess']:.1f}"
    )
    print("\nsrc/models.rs embeds this at compile time; rebuild to pick it up.")


def main() -> None:
    """Export a checkpoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", default="randles", help="which trained circuit")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--models-dir", type=Path, default=Path("src/models"))
    p.add_argument("--dtype", default="f16", choices=sorted(serialize_weights.DTYPES))
    args = p.parse_args()

    checkpoint = args.checkpoint or Path("training/checkpoints") / args.circuit / "best.pt"
    export(checkpoint, args.models_dir / f"{args.circuit}.eisnn", args.dtype)


if __name__ == "__main__":
    main()

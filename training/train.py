"""Train the param-guessing network.

The primary loss is the modulus-weighted residual of guess vs real curve.
A parameter negative log-likelihood (NLL) term is blended in with a decaying
weight, because residual loss alone can have plateaus. The NLL term also keeps
the log-variance head trained.

Set --lambda0 0 --warmup 0 to train on pure residual loss to check if NLL helps.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import circuits, dataset, evaluate, features, model, randles_torch, scales

ALPHA_RANGE = circuits.ALPHA_RANGE

# lambda never reaches zero, so the NLL keeps a gradient on out-of-range alpha
LAMBDA_FLOOR = 0.02


def to_params(targets: Tensor) -> Tensor:
    """Standardised-space targets -> normalised physical parameters, differentiably."""
    out = torch.empty_like(targets)
    for i in range(circuits.N_PARAMS):
        if i in circuits.LOG_PARAMS:
            out[..., i] = torch.pow(10.0, targets[..., i])
        else:
            out[..., i] = targets[..., i].clamp(*ALPHA_RANGE)
    return out


def nll(mu: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
    """Heteroscedastic Gaussian negative log-likelihood."""
    return (0.5 * (mu - target) ** 2 / log_var.exp() + 0.5 * log_var).mean()


def lambda_at(step: int, lambda0: float, decay_steps: int) -> float:
    """Exponential decay from lambda0 down to LAMBDA_FLOOR."""
    if lambda0 <= 0.0:
        return 0.0
    factor = np.exp(-step / max(decay_steps, 1))
    return float(max(lambda0 * factor, LAMBDA_FLOOR))


class Standardiser:
    """Applies and undoes the per-target mean/std."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, device: torch.device) -> None:
        self.mean_np, self.std_np = mean, std
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)

    def encode(self, targets: Tensor) -> Tensor:
        return (targets - self.mean) / self.std

    def decode(self, standardised: Tensor) -> Tensor:
        return standardised * self.std + self.mean


def guess(
    net: model.RandlesNet,
    std: Standardiser,
    device: torch.device,
    freqs: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Physical starting parameters from the torch network.

    Mirrors `Guesser::guess` in src/nn.rs; tests/test_parity.py holds the two
    together, so a change here needs the same change there.
    """
    net.eval()
    with torch.no_grad():
        f = features.extract(freqs, z, scales.DEFAULT)
        grid = torch.tensor(f.grid[None], dtype=torch.float32, device=device)
        scalars = torch.tensor(f.scalars[None], dtype=torch.float32, device=device)
        mu, _ = net(grid, scalars)

    targets = std.decode(mu).cpu().numpy()[0].astype(np.float64)
    normalised = circuits.from_targets(targets[None])
    normalised[0, circuits.ALPHA] = np.clip(normalised[0, circuits.ALPHA], *ALPHA_RANGE)
    return circuits.to_physical(normalised, f.k, f.w_c)[0]


def make_guess_init_params(
    net: model.RandlesNet, std: Standardiser, device: torch.device
):
    """Wrap the network as an evaluate.InitParams."""

    def guess_init_params(spectrum):
        return guess(net, std, device, spectrum.freqs, spectrum.z)

    return guess_init_params


def load_checkpoint(
    path: Path, device: torch.device | None = None
) -> tuple[model.RandlesNet, Standardiser, torch.device]:
    """Rebuild a trained network from a checkpoint, for evaluation and parity tests."""
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    if "config" not in payload:
        raise ValueError(f"{path} predates the architecture field; retrain it")
    config = model.Config(**payload["config"])
    net = model.RandlesNet(config).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    std = Standardiser(payload["target_mean"], payload["target_std"], device)
    return net, std, device


def evaluate_model(net, std, device, spectra, floor) -> evaluate.Summary:
    """Score the current network on the held-out set with plain LM."""
    outcomes = evaluate.fit_all(spectra, make_guess_init_params(net, std, device))
    net.train()
    return evaluate.summarise("model", outcomes, floor)


def main() -> None:
    """Train and checkpoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=300, help="pure parameter-MSE steps")
    p.add_argument("--lambda0", type=float, default=1.0)
    p.add_argument("--lambda-decay", type=int, default=3000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-n", type=int, default=300)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--channels", type=int, default=model.DEFAULT.channels)
    p.add_argument("--blocks", type=int, default=model.DEFAULT.blocks)
    p.add_argument("--head-width", type=int, default=model.DEFAULT.head_width)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("training/checkpoints/randles"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    print("computing target statistics...")
    mean, std_dev = dataset.target_statistics()
    std = Standardiser(mean, std_dev, device)
    print(f"  target mean {np.round(mean, 3)}")
    print(f"  target std  {np.round(std_dev, 3)}")

    config = model.Config(
        channels=args.channels, blocks=args.blocks, head_width=args.head_width
    )
    net = model.RandlesNet(config).to(device)
    print(f"model {config}: {model.parameter_count(net):,} parameters")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    loader = DataLoader(
        dataset.RandlesStream(seed=args.seed),
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    eval_spectra = evaluate.validation_set(args.eval_n)
    eval_floor = evaluate.fit_all(eval_spectra, evaluate.truth_init_params)

    best = float("inf")
    start = time.perf_counter()
    stream = iter(loader)

    for step in range(1, args.steps + 1):
        grid, scalars, targets = (t.to(device) for t in next(stream))
        target_std = std.encode(targets)

        mu, log_var = net(grid, scalars)
        param_term = nll(mu, log_var, target_std)

        if step <= args.warmup:
            loss = param_term
            lam = 1.0
            residual_term = torch.zeros((), device=device)
        else:
            w, z = randles_torch.spectrum_from_grid(grid.double())
            params = to_params(std.decode(mu).double())
            residual_term = randles_torch.residual_loss(params, w, z).mean()
            lam = lambda_at(step - args.warmup, args.lambda0, args.lambda_decay)
            loss = residual_term + lam * param_term

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0:
            print(
                f"step {step:>6}  loss {loss.item():>9.4f}  "
                f"residual {residual_term.item():>9.4f}  nll {param_term.item():>8.4f}  "
                f"lambda {lam:>6.3f}  {time.perf_counter() - start:>6.0f}s"
            )

        if step % args.eval_every == 0 or step == args.steps:
            summary = evaluate_model(net, std, device, eval_spectra, eval_floor)
            print(
                f"  eval  converged {100 * summary.converged:5.1f}%  "
                f"excess med {summary.median_excess:6.1f}  p90 {summary.p90_excess:6.1f}"
            )
            # Cost only matters if gate passes (i.e. it converges correctly)
            # It doesn't matter if convergense is fast if it is wrong
            score = (1.0 - summary.converged) * 1e6 + summary.median_excess
            if score < best:
                best = score
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "config": config.as_dict(),
                        "target_mean": mean,
                        "target_std": std_dev,
                        "estimator": scales.DEFAULT,
                        "step": step,
                        "converged": summary.converged,
                        "median_excess": summary.median_excess,
                    },
                    args.out / "best.pt",
                )
                print(f"  saved (score {score:.2f})")

    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()

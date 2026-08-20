# Copyright © 2026, Empa.
"""Resuming a run must not inherit the previous schedule's exhausted learning rate."""

from __future__ import annotations

import torch

from training.train import restore_optimizer

LR = 3e-4


def _annealed_state(steps: int = 20) -> tuple[dict, float]:
    """Optimizer state saved at the end of a cosine schedule, plus its final lr."""
    param = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([param], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        param.grad = torch.ones_like(param)
        opt.step()
        sched.step()
    return opt.state_dict(), opt.param_groups[0]["lr"]


def test_finished_schedule_has_low_learning_rate() -> None:
    """Finished state should have ~0.0 lr."""
    _, final_lr = _annealed_state()
    assert final_lr < 1e-9


def test_restoring_learning_rate() -> None:
    """Restoring checkout resets learning rate."""
    state, _ = _annealed_state()
    param = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([param], lr=LR)

    restore_optimizer(opt, state, LR)

    assert opt.param_groups[0]["lr"] == LR
    assert "initial_lr" not in opt.param_groups[0]


def test_resumed_optimizer_moves_weights() -> None:
    """Restored training moves the weights."""
    state, _ = _annealed_state()
    param = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([param], lr=LR)
    restore_optimizer(opt, state, LR)

    before = param.detach().clone()
    param.grad = torch.ones_like(param)
    opt.step()

    assert not torch.equal(before, param.detach())


def test_a_fresh_schedule_starts_from_the_requested_rate() -> None:
    """CosineAnnealingLR built after the restore must anneal from lr, not from 0."""
    state, _ = _annealed_state()
    param = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([param], lr=LR)
    restore_optimizer(opt, state, LR)

    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
    assert opt.param_groups[0]["lr"] == LR

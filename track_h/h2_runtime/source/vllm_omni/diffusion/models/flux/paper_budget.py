# SPDX-License-Identifier: Apache-2.0
"""Consumable velocity-deviation budget from full_method/current_v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch


EPSILON = 1e-8
ALPHA = 1.5
PROFILE_BUDGETS = {"aggressive": 0.02, "conservative": 0.01}


@dataclass(frozen=True)
class BudgetDecision:
    step: int
    action: str
    score: float
    balance_before: float
    balance_after: float
    ratio: float | None
    reason: str

    @property
    def compute_transformer(self) -> bool:
        return self.action != "reuse"

    @property
    def accumulate(self) -> float:
        return self.balance_after


class TensorBudgetController:
    """Choose one action from each pre-update latent and two used velocities."""

    def __init__(self, sigmas):
        self.sigmas = tuple(float(value) for value in sigmas)
        if len(self.sigmas) != 29 or any(not math.isfinite(value) for value in self.sigmas):
            raise ValueError("29 finite normalized native sigmas required")
        if not (0 <= self.sigmas[-1] < self.sigmas[0] <= 1) or any(
            before <= after for before, after in zip(self.sigmas, self.sigmas[1:])
        ):
            raise ValueError("native sigmas must strictly decrease within [0,1]")
        self.ratios = (1.0,) + tuple(1 + self.sigmas[i] - self.sigmas[i - 1] for i in range(1, 28))
        if any(not 0 < ratio <= 1 for ratio in self.ratios):
            raise ValueError("paper scaling factor outside (0,1]")
        self.initial_budget = self.balance = 0.0
        self.next_step = 0
        self.steps: list[BudgetDecision] = []
        self.profile: str | None = None
        self.hit_steps = 0

    def bind_profile(self, profile: str) -> None:
        if self.next_step != 8 or self.profile is not None or profile not in PROFILE_BUDGETS:
            raise ValueError("single routed profile binding requires completed actual W8")
        self.profile = profile
        self.initial_budget = self.balance = PROFILE_BUDGETS[profile]

    def score(self, step, latent, history) -> float:
        if self.steps[-1].action == "reuse":
            return ALPHA * self.steps[-1].score
        if len(history) < 2:
            raise ValueError("two complete used velocities required")
        older, previous = history[-2:]
        if previous.shape != latent.shape or older.shape != latent.shape:
            raise ValueError("complete history must match latent")
        residual = previous.float() - self.ratios[step - 1] * older.float()
        score = (
            abs(self.sigmas[step + 1] - self.sigmas[step])
            * torch.linalg.vector_norm(residual)
            / (torch.linalg.vector_norm(latent.float()) + EPSILON)
        )
        value = float(score.detach().cpu())
        if not math.isfinite(value) or value < 0:
            raise ValueError("nonfinite reuse score")
        return value

    def decide(self, step, latent, history) -> BudgetDecision:
        if type(step) is not int or step != self.next_step or not 0 <= step < 28:
            raise ValueError("controller requires exactly consecutive steps 0..27")
        if step >= 8 and self.profile is None:
            raise ValueError("paper budget requires a routed profile after actual W8")
        before = self.balance
        if step < 8 or step == 23:
            action, score = "dense", 0.0
            reason = "warmup" if step < 8 else "refresh23"
        else:
            score = self.score(step, latent, history)
            if score <= before:
                action, reason = "reuse", "budget_accept"
            else:
                action, reason = "sparse", "budget_exhausted"
        after = before - score if action == "reuse" else self.initial_budget
        result = BudgetDecision(step, action, score, before, after, self.ratios[step] if action == "reuse" else None, reason)
        self.balance = after
        self.next_step += 1
        self.steps.append(result)
        self.hit_steps += int(action == "reuse")
        return result

    def metadata(self):
        body = {
            "schema_version": 1,
            "kind": "paper_consumable_velocity_budget",
            "profile": self.profile,
            "B": self.initial_budget,
            "alpha": ALPHA,
            "epsilon": EPSILON,
            "sigmas": list(self.sigmas),
            "ratios": list(self.ratios),
            "r_definition": "1+sigma_i-sigma_(i-1)",
            "score_arithmetic": "FP32 operands, Frobenius norms over complete generated velocity/latent",
            "velocity_source": "immediate_previous_complete_scheduler_velocity",
            "prediction_cache": "unchanged on reuse; sparse merges into last computed per-position predictions",
            "steps": [asdict(row) for row in self.steps],
            "action_counts": {action: sum(row.action == action for row in self.steps) for action in ("dense", "sparse", "reuse")},
        }
        body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return body

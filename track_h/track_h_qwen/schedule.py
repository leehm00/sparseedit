# SPDX-License-Identifier: Apache-2.0
"""Calibration and two-repetition fixed-trajectory H1 schedules."""

from dataclasses import dataclass

CALIBRATION_BUDGETS = (20, 10, 35)
MEASUREMENT_ARMS = (
    ("direct_sparse", "direct", False),
    ("ready_eager", "ready", False),
    ("full_sparsegraph", "ready", True),
)
ADMISSION_ARMS = (
    ("admit_direct", "direct", False),
    ("admit_ready", "ready", False),
    ("admit_graph", "ready", True),
)


@dataclass(frozen=True)
class Job:
    case: dict
    arm: str
    mode: str
    budget: int | None = None
    graph: bool = False
    repetition: int | None = None
    admission: bool = False

    @property
    def identity(self):
        return self.case["case_id"], self.arm, self.repetition


def collect(case, budget):
    return Job(case, f"direct_collect{budget}", "direct", budget)


def calibration_jobs(cases):
    jobs = [Job(cases[0], "warmup", "dense")]
    jobs.extend(Job(case, "fresh_dense", "dense", repetition=0) for case in cases)
    for budget in (*CALIBRATION_BUDGETS, 40):
        for index, case in enumerate(cases if budget in CALIBRATION_BUDGETS else cases[:1]):
            jobs.append(collect(case, budget))
            if index == 0:
                jobs.extend(Job(case, f"{arm}{budget}", mode, budget, graph, admission=True)
                            for arm, mode, graph in ADMISSION_ARMS)
            jobs.extend(Job(case, f"{arm}{budget}", mode, budget, graph, repetition=0)
                        for arm, mode, graph in MEASUREMENT_ARMS)
    return jobs


def heldout_jobs(cases):
    producers = [collect(case, 20) for case in cases]
    repeats = []
    for repetition in (0, 1):
        cells = []
        for case in cases:
            cells.append(Job(case, "fresh_dense", "dense", repetition=repetition))
            cells.extend(Job(case, f"{arm}20", mode, 20, graph, repetition)
                         for arm, mode, graph in MEASUREMENT_ARMS)
        repeats.extend(cells if repetition == 0 else reversed(cells))
    return producers + repeats

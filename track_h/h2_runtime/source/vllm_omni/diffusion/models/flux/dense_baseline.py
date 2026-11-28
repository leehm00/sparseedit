# SPDX-License-Identifier: Apache-2.0
"""Static graph capture for the unchanged native FLUX Dense forward."""

import torch

from .specialized_graph import BranchSparseGraph


class DenseBaselineGraphs:
    def __init__(self):
        self.arguments = {}
        self.graphs = {}
        self.pool = None

    def bind(self, branch, source):
        if branch not in self.arguments:
            self.arguments[branch] = {
                name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for name, value in source.items()
            }
            return self.arguments[branch]
        owned = self.arguments[branch]
        if owned.keys() != source.keys():
            raise ValueError("Dense graph static argument set changed")
        for name, value in source.items():
            saved = owned[name]
            if isinstance(value, torch.Tensor) and isinstance(saved, torch.Tensor):
                if (saved.shape, saved.dtype, saved.device) != (value.shape, value.dtype, value.device):
                    raise ValueError(f"Dense graph {name} layout changed")
                saved.copy_(value)
            elif saved != value:
                raise ValueError(f"Dense graph {name} static argument changed")
        return owned

    def forward(self, branch, predict, kwargs):
        owned = self.bind(branch, kwargs)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        graph = self.graphs.get(branch)
        if graph is None:
            graph = BranchSparseGraph(
                branch, borrow_output=True, shared_pool=self.pool,
                captured_region="native Dense FLUX transformer",
            )
            graph.capture(lambda: predict(**owned), device=owned["hidden_states"].device)
            if branch == "negative" and self.graphs["positive"].output_storage_ptr() == graph.output_storage_ptr():
                raise RuntimeError("Dense graph CFG output slots alias")
            self.graphs[branch] = graph
        return graph.replay()

    def replay_counts(self):
        return {name: graph.replay_count for name, graph in self.graphs.items()}

    def metadata(self, before):
        return {
            "active": True,
            "branches": {name: g.metadata() for name, g in self.graphs.items()},
            "request_replays": {name: count - before.get(name, 0) for name, count in self.replay_counts().items()},
            "request_captured": not bool(before),
        }

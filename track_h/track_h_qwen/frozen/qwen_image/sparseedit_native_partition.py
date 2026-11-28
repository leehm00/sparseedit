# SPDX-License-Identifier: Apache-2.0
"""Opt-in eager native Qwen block dispatch and allocation-free capacity contracts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import torch


def original_partition_assets(model):
    """Gate complete local original components and freeze exact content identity.

    Full hashes are read only after the cheap header capacity gate admits the
    topology. The canonical upstream revision is a separate acquisition input;
    this manifest identifies the exact bytes actually loaded by the request.
    """
    root = Path(model)
    index_path = root / "model_index.json"
    if not index_path.is_file():
        raise ValueError("original partition snapshot requires model_index.json and all native components")
    index = json.loads(index_path.read_text())
    if index.get("_class_name") != "QwenImageEditPipeline" or "2509" in root.name:
        raise ValueError("original partition snapshot must declare QwenImageEditPipeline")
    files = [index_path]
    for name in ("transformer", "text_encoder", "vae", "tokenizer", "processor", "scheduler"):
        directory = root / name
        component = sorted(p for p in directory.rglob("*") if p.is_file() and not any(v.startswith(".") for v in p.relative_to(directory).parts))
        if not component or name not in index:
            raise ValueError(f"original partition snapshot missing component: {name}")
        if name in ("transformer", "text_encoder", "vae") and not any(p.suffix == ".safetensors" for p in component):
            raise ValueError(f"original partition snapshot requires complete safetensors: {name}")
        for shard_index in directory.glob("*.safetensors.index.json"):
            mapping = json.loads(shard_index.read_text())["weight_map"]
            if any(not (directory / shard).is_file() for shard in mapping.values()):
                raise ValueError(f"original partition snapshot missing indexed shard: {name}")
        files.extend(component)
    manifest = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return {"files": manifest, "snapshot_content_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            "canonical_revision": index.get("_commit_hash"), "canonical_revision_scope": "declared_metadata_requires_acquisition_verification",
            "identity_scope": "all_native_component_file_contents_actually_supplied_to_local_only_loader"}


@dataclass(frozen=True)
class QwenPartitionPlan:
    layers: int
    devices: tuple[str, str]
    split: int
    device_ids: tuple[str, str]

    @classmethod
    def create(cls, layers, devices, split=None, device_ids=None):
        values = devices.split(",") if isinstance(devices, str) else list(devices)
        try:
            parsed = tuple(f"cuda:{int(str(v).removeprefix('cuda:'))}" for v in values)
        except ValueError as exc:
            raise ValueError("partition requires two explicit CUDA ordinals") from exc
        split = layers // 2 if split is None else split
        if len(parsed) != 2 or parsed[0] == parsed[1] or any(int(v.split(":")[1]) < 0 for v in parsed):
            raise ValueError("partition requires two distinct nonnegative CUDA devices")
        if layers < 2 or not 0 < split < layers:
            raise ValueError("partition requires two nonempty contiguous block stages")
        ids = tuple(device_ids or parsed)
        if len(ids) != 2 or len(set(ids)) != 2:
            raise ValueError("partition requires two distinct device identities")
        return cls(int(layers), parsed, int(split), ids)

    def layer_device(self, layer):
        if not 0 <= layer < self.layers:
            raise ValueError("layer outside partition coverage")
        return self.devices[int(layer >= self.split)]

    @property
    def device_map(self):
        result = {name: self.devices[0] for name in ("img_in", "txt_norm", "txt_in", "time_text_embed", "pos_embed")}
        result.update({f"transformer_blocks.{i}": self.layer_device(i) for i in range(self.layers)})
        result.update({name: self.devices[1] for name in ("norm_out", "proj_out")})
        return result

    def owner(self, parameter):
        for name, device in self.device_map.items():
            if parameter == name or parameter.startswith(name + "."):
                return device
        raise ValueError(f"unowned transformer tensor: {parameter}")

    @property
    def payload(self):
        return {"layers": self.layers, "split": self.split, "devices": list(self.devices),
                "device_ids": list(self.device_ids), "device_map": self.device_map,
                "request_scheduler_output_device": self.devices[0], "execution": "eager_native_accelerate_block_dispatch"}

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.payload, sort_keys=True).encode()).hexdigest()


def partition_capacity(model, plan, *, generated_tokens, reference_tokens, available_bytes,
                       text_lengths=None, workspace_bytes=2 * 1024**3, cache_kv=True):
    """Inspect original local headers; do not allocate tensors or initialize CUDA."""
    directory = Path(model) / "transformer"
    config = json.loads((directory / "config.json").read_text())
    if int(config["num_layers"]) != plan.layers:
        raise ValueError("partition coverage differs from actual transformer config")
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError("partition preflight requires actual safetensors headers")
    weights = dict.fromkeys(plan.devices, 0)
    tensors, manifest, seen_layers = {}, [], set()
    dtype_bytes = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                   "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    for path in files:
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError("truncated safetensors header")
            size = struct.unpack("<Q", raw)[0]
            if size > 64 * 1024**2 or size + 8 > path.stat().st_size:
                raise ValueError("invalid safetensors header size")
            header_raw = handle.read(size)
            header = json.loads(header_raw)
        # Header identity and length describe the capacity input. Full weight hashes
        # belong to the snapshot manifest, not a costly capacity read.
        manifest.append({"file": path.name, "file_bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(raw + header_raw).hexdigest(), "hash_scope": "length_and_header"})
        for name, row in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate safetensors tensor: {name}")
            count = math.prod(row["shape"])
            start, stop = row["data_offsets"]
            if row["dtype"] not in dtype_bytes or stop - start != count * dtype_bytes[row["dtype"]] or not 0 <= start <= stop <= path.stat().st_size - 8 - size:
                raise ValueError(f"invalid safetensors tensor extent: {name}")
            tensors[name] = path.name
            weights[plan.owner(name)] += count * 2  # loader explicitly requests BF16
            if name.startswith("transformer_blocks."):
                seen_layers.add(int(name.split(".")[1]))
    if seen_layers != set(range(plan.layers)):
        raise ValueError("actual header block coverage differs from partition")
    for index in directory.glob("*.safetensors.index.json"):
        mapping = json.loads(index.read_text())["weight_map"]
        if mapping != tensors:
            raise ValueError("safetensors index/header coverage differs")
    if generated_tokens <= 0 or reference_tokens <= 0 or workspace_bytes < 0:
        raise ValueError("positive generated/reference geometry and nonnegative reserve required")
    if text_lengths is not None and (len(text_lengths) != 2 or any(n <= 0 for n in text_lengths)):
        raise ValueError("capacity requires both actual unequal-or-equal CFG text lengths")
    width = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    rows = []
    for stage, device in enumerate(plan.devices):
        layers = plan.split if stage == 0 else plan.layers - plan.split
        image = 2 * 2 * layers * width * (generated_tokens + reference_tokens) * 2 if cache_kv else 0
        text = 2 * layers * width * sum(text_lengths or []) * 2 if cache_kv else 0
        required = weights[device] + image + text + workspace_bytes
        rows.append({"device": device, "device_id": plan.device_ids[stage], "layers": layers,
                     "resident_transformer_bytes": weights[device], "two_cfg_image_kv_bytes": image,
                     "ready_text_kv_bytes": text, "workspace_reserve_bytes": workspace_bytes,
                     "required_bytes": required, "available_bytes": available_bytes[device],
                     "status": "blocked_capacity" if required > available_bytes[device] else "admitted"})
    return {"status": "blocked_capacity" if any(r["status"] == "blocked_capacity" for r in rows) else
            "admitted_exact_context_estimate" if text_lengths is not None else "admitted_lower_bound_pending_native_text_length_gate",
            "stages": rows, "partition": plan.payload, "partition_sha256": plan.identity,
            "model_config_sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
            "header_manifest": manifest, "text_lengths": text_lengths, "persistent_cfg_cache": cache_kv,
            "generated_tokens": generated_tokens, "reference_tokens": reference_tokens,
            "workspace_reserve_bytes": workspace_bytes, "dtype": "bfloat16",
            "scope": "actual_header_stage_weights_two_CFG_image_and_ready_text_KV_explicit_workspace"}


class LocalTensorCache:
    """One copy per source/version/device per call, stable optional stage slots."""
    def __init__(self):
        self.memo, self.buffers, self.transfers = {}, {}, []

    def begin(self):
        self.memo.clear()

    def local(self, value, device, *, slot=None):
        if not isinstance(value, torch.Tensor):
            return value
        device = torch.device(device)
        if value.device == device:
            return value
        # Strong source reference in the memo prevents Python id reuse.
        key = (id(value), value._version if not torch.is_inference(value) else None, str(device), slot)
        if key in self.memo:
            return self.memo[key][1]
        buffer_key = (slot, str(device))
        previous = self.buffers.get(buffer_key) if slot is not None else None
        if previous is not None and (previous.shape, previous.dtype) == (value.shape, value.dtype):
            previous.copy_(value)
            result = previous
        else:
            result = value.to(device)
            if slot is not None:
                self.buffers[buffer_key] = result
        self.memo[key] = (value, result)
        self.transfers.append({"source": str(value.device), "target": str(device),
                               "bytes": value.numel() * value.element_size(), "slot": slot,
                               "scope": "explicit_tensor_copy"})
        return result

    def tree(self, value, device, path=""):
        if isinstance(value, dict):
            return {k: self.tree(v, device, path + "/" + k) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self.tree(v, device, path + f"/{i}") for i, v in enumerate(value))
        # Hidden/text streams change at every block; never alias them into shared slots.
        slot = path if any(name in path for name in ("rotary", "temb", "mask", "modulate")) else None
        return self.local(value, device, slot=slot)


def synchronize_devices(model):
    plan = getattr(model, "_sparseedit_partition", None)
    devices = plan.devices if plan is not None else (next(model.parameters()).device,)
    for device in devices:
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)


def dispatch_partition(model, plan):
    """Keep installed native forward; only argument ownership/output device changes."""
    from accelerate import dispatch_model

    if len(model.transformer_blocks) != plan.layers:
        raise ValueError("model block count differs from partition")
    for name, _ in model.named_parameters():
        plan.owner(name)
    local = LocalTensorCache()
    model._sparseedit_partition = plan
    model._sparseedit_transfer_cache = local
    model = dispatch_model(model, plan.device_map, main_device=plan.devices[0], force_hooks=True)
    handles = []
    for index, block in enumerate(model.transformer_blocks):
        def prepare(module, args, kwargs, index=index):
            if index == 0:
                local.begin()
            return local.tree(args, plan.layer_device(index), "args"), local.tree(kwargs, plan.layer_device(index), "kwargs")
        handles.append(block.register_forward_pre_hook(prepare, with_kwargs=True))
    # Accelerate's root io_same_device returns to request/scheduler device0.
    model._sparseedit_partition_handles = handles
    return model

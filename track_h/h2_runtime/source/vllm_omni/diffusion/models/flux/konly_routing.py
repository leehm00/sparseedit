# SPDX-License-Identifier: Apache-2.0
"""One-prefix W8 K-only routing. No temporal-reuse controller is implemented here."""
from __future__ import annotations
import math
import torch

def choose_profile(score):
    if not math.isfinite(float(score)):raise ValueError('nonfinite K-only score')
    return 'aggressive' if score>.5 else 'conservative'

class CleanPredictionWarmupState:
    def __init__(self):
        self.previous=None
        self.d=[]
        self.next_step=0
        self.reference_checked=False

    def observe(self,step,z,velocity,sigma,reference,generated_ids,reference_ids):
        if step!=self.next_step or not 0<=step<8:raise ValueError('warmup sequence must be 0..7')
        if not self.reference_checked:
            # Kontext uses stream id 0 for generated and 1 for reference; the
            # spatial coordinates must match exactly and channel order is shared
            # by the native _pack_latents preprocessing of both streams.
            if z.shape!=reference.shape or velocity.shape!=z.shape or generated_ids.shape!=reference_ids.shape:
                raise ValueError('aligned reference geometry differs')
            if not torch.equal(generated_ids[...,1:],reference_ids[...,1:]):
                raise ValueError('aligned reference position geometry differs')
            if not bool(torch.isfinite(reference).all()):raise ValueError("nonfinite aligned reference")
            self.reference_checked=True
        # Convert operands before multiply/subtract, including the actual native
        # scheduler sigma; BF16 subtraction followed by float() is different.
        sigma32=torch.as_tensor(sigma,device=z.device,dtype=torch.float32)
        clean=z.float()-sigma32*velocity.float()
        if self.previous is not None:self.d.append(torch.linalg.vector_norm(clean-self.previous,dim=-1).mean())
        self.previous=clean
        self.next_step+=1
        if step==7:self.coverage=(torch.linalg.vector_norm(clean-reference.float(),dim=-1)>3.).float().mean()

    def finish(self):
        if self.next_step!=8:raise ValueError('exactly eight actual warmup observations required')
        late=(self.d[5]+self.d[6])/2
        convergence=late/((self.d[0]+self.d[1])/2+1e-8)
        components=torch.stack((1/(1+late),1/(1+convergence),1-self.coverage))
        # A single explicit host read establishes the online dispatch decision.
        values=torch.cat((torch.stack(self.d),torch.stack((late,convergence,self.coverage)),components,components.mean().reshape(1))).detach().cpu().tolist()
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite K-only score')
        score=values[-1]
        return dict(d=values[:7],L=values[7],C=values[8],A=values[9],components=values[10:13],score=score,
                    profile=choose_profile(score),epsilon=1e-8,eta=3.,theta=.5,arithmetic='fp32 operands/multiply/subtract/norm/mean')

def bind_selected_template(pipeline, owner, k, *, timesteps, config):
    """Transfer one request to a K-specific template with the W8 KV arena."""
    from .cfg_execution import CFGExecutionPlan
    from .condition_projection_plan import RequestConditionPlan
    from .request_positions import RequestPositionPlan
    from .specialized_graph import BranchSparseGraph, RequestGraphState
    from .specialized_plan import SpecializedExecutionPlan
    from .specialized_template import bind_template_caches, reserve_template

    if not owner.in_use or owner.closed:
        raise RuntimeError("K-only binding needs an active serial request")
    if type(k) is not int or not 1 <= k <= owner.plan.generated_token_count:
        raise ValueError("invalid selected K")
    for cache in (owner.positive_cache, owner.negative_cache):
        if cache is None or not cache.layers:
            raise RuntimeError("K-only binding requires actual W8 full branch KV")

    owner_cfg = owner.cfg_plan
    joint_cache = None
    if owner_cfg is not None:
        joint_cache = owner_cfg.joint_cache
        if joint_cache is None or owner.arena is None or joint_cache is not owner.arena.joint_cache:
            raise RuntimeError("K-only binding requires the arena-owned joint CFG cache")
        for layer in owner.positive_cache.layers.keys() | owner.negative_cache.layers.keys():
            owner_cfg.pool.require_full(layer, None)

    if owner.plan.active_token_count == k:
        return owner, owner.graphs.positive.graph_active
    signature = owner.signature[:4] + (k,) + owner.signature[5:]
    selected, _ = reserve_template(pipeline, owner, signature)
    original = owner.plan
    try:
        if selected.full_img_ids is None:
            selected.full_img_ids = owner.full_img_ids.detach().clone()
        else:
            selected.full_img_ids.copy_(owner.full_img_ids)
        for branch, tensors in owner.conditioning.items():
            selected.bind_conditioning(branch, tensors)
        positive, negative = bind_template_caches(
            pipeline,
            selected,
            owner.arena_signature,
            owner.positive_cache,
            owner.negative_cache,
            joint_cache=joint_cache,
        )
        if positive is not owner.positive_cache or negative is not owner.negative_cache:
            raise RuntimeError("K-only cache arena ownership differs")
        selected.positive_cache, selected.negative_cache = positive, negative

        if selected.plan is None:
            selected.plan = SpecializedExecutionPlan(
                variant="graph_reuse",
                generated_token_count=original.generated_token_count,
                reference_token_count=original.reference_token_count,
                active_token_count=k,
                latent_channels=original.latent_channels,
                dtype=original.dtype,
                device=original.device,
                generated_ids=selected.full_img_ids[:original.generated_token_count],
                fuse_stage=original.fuse_stage,
            )
            selected.position_plan = RequestPositionPlan(
                text_tokens=owner.position_plan.text_tokens,
                image_tokens=owner.position_plan.image_tokens,
                active_tokens=k,
                dtype=original.dtype,
                device=original.device,
                share_branches=owner.position_plan.share_branches,
            )
            selected.condition_plan = RequestConditionPlan(
                mode="reuse", cache=pipeline._sparseedit_condition_bank
            )
            graph_options = {
                "allow_fallback": config.sparseedit_graph_fallback,
                "warmup_trials": config.sparseedit_graph_warmup_trials,
            }
            selected.graphs = RequestGraphState(
                positive=BranchSparseGraph(
                    "cfg_batched" if joint_cache is not None else "positive", **graph_options
                ),
                negative=BranchSparseGraph("negative", **graph_options) if joint_cache is None else None,
                allow_fallback=config.sparseedit_graph_fallback,
            )
        if joint_cache is not None:
            if selected.cfg_plan is None:
                selected.cfg_plan = CFGExecutionPlan(
                    active_tokens=k,
                    latent_channels=original.latent_channels,
                    dtype=original.dtype,
                    device=original.device,
                    joint_cache=joint_cache,
                )
            if selected.cfg_plan.joint_cache is not joint_cache:
                raise RuntimeError("K-only joint CFG cache arena ownership differs")
            pos, neg = selected.conditioning["positive"], selected.conditioning["negative"]
            selected.cfg_plan.bind_request(
                pos["prompt"], neg["prompt"], pos["pooled"], neg["pooled"], pos.get("guidance")
            )
        selected.position_plan.bind(
            pipeline.transformer.pos_embed,
            selected.full_img_ids,
            {branch: tensors["ids"] for branch, tensors in selected.conditioning.items()},
        )
        selected.condition_plan.bind(
            pipeline.transformer,
            timesteps,
            {branch: tensors["pooled"] for branch, tensors in selected.conditioning.items()},
            guidance=selected.conditioning["positive"].get("guidance"),
            dtype=original.dtype,
        )
    except BaseException as error:
        try:
            selected.finish(success=False)
        except BaseException as cleanup_error:
            error.add_note(f"Selected template cleanup failed: {cleanup_error}")
        raise
    selected.plan.bind_count = 0
    selected.in_use = True
    owner.in_use = False
    pipeline._sparseedit_forward_template = selected
    return selected, selected.graphs.positive.graph_active

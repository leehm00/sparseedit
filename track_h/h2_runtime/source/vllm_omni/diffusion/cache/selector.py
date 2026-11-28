from collections.abc import Mapping
from typing import Any

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
from vllm_omni.diffusion.cache.magcache import MagCacheBackend
from vllm_omni.diffusion.cache.stepcache import StepCacheBackend
from vllm_omni.diffusion.cache.teacache import TeaCacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig

TEA_CACHE_REQUEST_CONFIG_KEYS = frozenset({"rel_l1_thresh"})
CACHE_DIT_REQUEST_CONFIG_KEYS = frozenset(
    {
        "Fn_compute_blocks",
        "Bn_compute_blocks",
        "max_warmup_steps",
        "residual_diff_threshold",
        "max_continuous_cached_steps",
        "enable_taylorseer",
        "taylorseer_order",
        "scm_steps_mask_policy",
        "scm_steps_policy",
    }
)


def project_cache_config_override(
    cache_backend: str | None,
    extra_args: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return request fields owned by the selected cache backend."""
    if not extra_args:
        return {}
    keys = {
        "tea_cache": TEA_CACHE_REQUEST_CONFIG_KEYS,
        "cache_dit": CACHE_DIT_REQUEST_CONFIG_KEYS,
    }.get(str(cache_backend or "").lower(), frozenset())
    return {key: extra_args[key] for key in keys if key in extra_args}


def refresh_cache_backend_for_request(
    backend: CacheBackend,
    pipeline: Any,
    *,
    cache_backend: str | None,
    num_inference_steps: int,
    extra_args: Mapping[str, Any] | None,
) -> None:
    """Refresh a cache backend with its request-owned configuration."""
    config_override = project_cache_config_override(cache_backend, extra_args)
    if config_override:
        backend.refresh(
            pipeline,
            num_inference_steps,
            config_override=config_override,
        )
        return
    backend.refresh(pipeline, num_inference_steps)


def get_cache_backend(cache_backend: str | None, cache_config: Any) -> CacheBackend | None:
    """Get cache backend instance based on cache_backend string.

    This is a selector function that routes to the appropriate backend implementation.
    - cache_dit: Uses CacheDiTBackend with enable()/refresh() interface
    - tea_cache: Uses TeaCacheBackend with enable()/refresh() interface
    - mag_cache: Uses MagCacheBackend with enable()/refresh() interface
    - step_cache: Uses StepCacheBackend for DreamZero velocity step skip

    Args:
        cache_backend: Cache backend name ("cache_dit", "tea_cache",
            "mag_cache", "step_cache", or None).
        cache_config: Cache configuration (dict or DiffusionCacheConfig instance).

    Returns:
        Cache backend instance if cache_backend is set, None otherwise.

    Raises:
        ValueError: If cache_backend is unsupported.
    """
    if cache_backend is None or cache_backend == "none":
        return None

    if isinstance(cache_config, dict):
        cache_config = DiffusionCacheConfig.from_dict(cache_config)

    if cache_backend == "cache_dit":
        return CacheDiTBackend(cache_config)
    elif cache_backend == "tea_cache":
        return TeaCacheBackend(cache_config)
    elif cache_backend == "mag_cache":
        return MagCacheBackend(cache_config)
    elif cache_backend in ("step_cache", "stepcache", "step_cache_dit"):
        return StepCacheBackend(cache_config)
    else:
        raise ValueError(
            f"Unsupported cache backend: {cache_backend}. "
            "Supported: 'cache_dit', 'tea_cache', 'mag_cache', 'step_cache'"
        )

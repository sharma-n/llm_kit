"""Composite AppConfig and YAML loader with ${VAR} interpolation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Self, get_args, get_origin, get_type_hints

import yaml

from llm_kit.config.batch import BatchConfig
from llm_kit.config.embed import EmbeddingConfig
from llm_kit.config.http import HTTPConfig
from llm_kit.config.image import ImageConfig
from llm_kit.config.llm import LLMConfig
from llm_kit.config.provider_batch import ProviderBatchConfig
from llm_kit.config.rate_limit import RateLimitConfig


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


@dataclass(slots=True)
class AppConfig:
    """Top-level config. Compose from YAML via ``AppConfig.from_yaml(path)``."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    http: HTTPConfig = field(default_factory=HTTPConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    provider_batch: ProviderBatchConfig = field(default_factory=ProviderBatchConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Top-level YAML at {path} must be a mapping")
        data = _interpolate_env(data)
        return _build_dataclass(cls, data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return _build_dataclass(cls, _interpolate_env(data))


def _interpolate_env(value: Any) -> Any:
    """Recursively replace ${VAR} / ${VAR:-default} inside string values."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(_replace_env_match, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _replace_env_match(match: re.Match[str]) -> str:
    var, default = match.group(1), match.group(2)
    resolved = os.environ.get(var)
    if resolved is not None:
        return resolved
    if default is not None:
        return default
    raise KeyError(f"Environment variable {var!r} referenced in config but not set")


def _build_dataclass[T](cls: type[T], data: dict[str, Any]) -> T:
    """Recursively construct a dataclass tree from a dict.

    Why: dacite is great but adds a runtime dep we don't otherwise need.
    Supports nested dataclasses, enums, tuples, and primitive collections.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key {key!r} for {cls.__name__}")
        kwargs[key] = _coerce(hints[key], value)
    return cls(**kwargs)  # type: ignore[call-arg]


def _coerce(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(annotation)

    if is_dataclass(annotation) and isinstance(value, dict):
        return _build_dataclass(annotation, value)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)

    if origin is tuple:
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        return tuple(_coerce(item_type, v) for v in value)

    if origin is list:
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        return [_coerce(item_type, v) for v in value]

    if origin is dict:
        args = get_args(annotation)
        v_type = args[1] if len(args) == 2 else Any
        return {k: _coerce(v_type, v) for k, v in value.items()}

    # Union (e.g. ``int | None``): try the first non-None arm that the value
    # can plausibly satisfy. For the simple cases we use, the value type is
    # unambiguous and bare assignment is correct.
    if origin is not None and get_args(annotation):
        for arm in get_args(annotation):
            if arm is type(None):
                continue
            try:
                return _coerce(arm, value)
            except (TypeError, ValueError):
                continue
        return value

    return value

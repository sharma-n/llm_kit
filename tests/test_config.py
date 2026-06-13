"""Tests for AppConfig loader: env interpolation, enums, nested dataclasses."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from llm_kit.config.app import AppConfig
from llm_kit.config.batch import BatchFailurePolicy
from llm_kit.config.image import ImageFailurePolicy
from llm_kit.config.llm import StructuredOutputStyle


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(dedent(body), encoding="utf-8")
    return path


def test_load_example_config_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cfg = AppConfig.from_yaml(Path("examples/config.yaml"))

    assert cfg.llm.base_url == "https://api.openai.com"  # default branch
    assert cfg.llm.model == "gpt-4o-mini"
    assert cfg.llm.structured_output_style is StructuredOutputStyle.OPENAI_JSON_SCHEMA
    assert cfg.rate_limit.requests_per_minute == 3000
    assert cfg.rate_limit.tokens_per_minute == 1500000
    assert cfg.image.on_failure is ImageFailurePolicy.RAISE
    assert cfg.batch.on_failure is BatchFailurePolicy.RECORD
    # nested dataclass: independent retry config for images
    assert cfg.image.retry.max_retries == 3
    assert cfg.http.llm_retry.max_retries == 5


def test_provider_batch_defaults_and_yaml_override(tmp_path: Path) -> None:
    # Defaults present without any YAML section.
    assert AppConfig().provider_batch.completion_window == "24h"
    assert AppConfig().provider_batch.batches_path == "/v1/batches"

    path = _write(
        tmp_path,
        """
        provider_batch:
          completion_window: 48h
          poll_interval_seconds: 5.0
          max_requests_per_batch: 1000
        """,
    )
    cfg = AppConfig.from_yaml(path)
    assert cfg.provider_batch.completion_window == "48h"
    assert cfg.provider_batch.poll_interval_seconds == 5.0
    assert cfg.provider_batch.max_requests_per_batch == 1000


def test_env_var_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "from-env")
    path = _write(
        tmp_path,
        """
        llm:
          model: ${LLM_MODEL}
        """,
    )
    cfg = AppConfig.from_yaml(path)
    assert cfg.llm.model == "from-env"


def test_env_var_default_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_VAR", raising=False)
    path = _write(
        tmp_path,
        """
        llm:
          model: ${MISSING_VAR:-fallback-model}
        """,
    )
    cfg = AppConfig.from_yaml(path)
    assert cfg.llm.model == "fallback-model"


def test_missing_env_var_without_default_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    path = _write(
        tmp_path,
        """
        llm:
          model: ${DEFINITELY_NOT_SET}
        """,
    )
    with pytest.raises(KeyError, match="DEFINITELY_NOT_SET"):
        AppConfig.from_yaml(path)


def test_env_var_in_yaml_comment_is_ignored(tmp_path: Path) -> None:
    """Regression: comments are not subject to interpolation.

    Why: interpolation happens after YAML parse, so ${VAR} inside a `#` line
    is invisible to the env lookup.
    """
    path = _write(
        tmp_path,
        """
        # don't expand ${SECRET_VAR} in this comment
        llm:
          model: fixed
        """,
    )
    cfg = AppConfig.from_yaml(path)  # must not raise
    assert cfg.llm.model == "fixed"


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        llm:
          model: m
        garbage:
          x: 1
        """,
    )
    with pytest.raises(ValueError, match="Unknown config key 'garbage'"):
        AppConfig.from_yaml(path)


def test_unknown_nested_key_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        llm:
          not_a_real_field: 1
        """,
    )
    with pytest.raises(ValueError, match="Unknown config key 'not_a_real_field'"):
        AppConfig.from_yaml(path)


def test_enum_coercion_from_string(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        image:
          on_failure: placeholder
        batch:
          on_failure: raise
        """,
    )
    cfg = AppConfig.from_yaml(path)
    assert cfg.image.on_failure is ImageFailurePolicy.PLACEHOLDER
    assert cfg.batch.on_failure is BatchFailurePolicy.RAISE


def test_tuple_field_coercion(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        http:
          llm_retry:
            retry_on_status: [429, 503]
        """,
    )
    cfg = AppConfig.from_yaml(path)
    assert cfg.http.llm_retry.retry_on_status == (429, 503)


def test_non_mapping_top_level_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        AppConfig.from_yaml(path)


def test_from_dict_also_interpolates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_MODEL", "dict-model")
    cfg = AppConfig.from_dict({"llm": {"model": "${MY_MODEL}"}})
    assert cfg.llm.model == "dict-model"

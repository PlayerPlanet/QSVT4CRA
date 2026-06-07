"""Tests for compiler_backend.heron.load_ibm_token.

The loader must resolve the IBM Quantum token from, in order:
    1. $IBM_API_KEY (or the named env var override)
    2. $IBM_API_KEY_FILE
    3. <token_file> defaulting to .ibm_token in cwd
and raise a clear error if none of these are populated.
"""
from __future__ import annotations

import pytest

from compiler_backend.heron import DEFAULT_TOKEN_FILE, load_ibm_token


def test_env_var_wins(monkeypatch, tmp_path):
    monkeypatch.delenv("IBM_API_KEY_FILE", raising=False)
    monkeypatch.setenv("IBM_API_KEY", "env-token-123")
    token = load_ibm_token(token_file=tmp_path / "should_not_be_read")
    assert token == "env-token-123"


def test_file_env_var(monkeypatch, tmp_path):
    token_file = tmp_path / "tok"
    token_file.write_text("file-token-xyz\n", encoding="utf-8")
    monkeypatch.delenv("IBM_API_KEY", raising=False)
    monkeypatch.setenv("IBM_API_KEY_FILE", str(token_file))
    token = load_ibm_token()
    assert token == "file-token-xyz"


def test_default_token_file(monkeypatch, tmp_path):
    monkeypatch.delenv("IBM_API_KEY", raising=False)
    monkeypatch.delenv("IBM_API_KEY_FILE", raising=False)
    token_file = tmp_path / DEFAULT_TOKEN_FILE
    token_file.write_text("default-file-token\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    token = load_ibm_token()
    assert token == "default-file-token"


def test_comments_and_blanks_ignored(monkeypatch, tmp_path):
    token_file = tmp_path / DEFAULT_TOKEN_FILE
    token_file.write_text(
        "# this is a comment\n\n   \nactual-token\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("IBM_API_KEY", raising=False)
    monkeypatch.delenv("IBM_API_KEY_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    token = load_ibm_token()
    assert token == "actual-token"


def test_env_var_beats_file(monkeypatch, tmp_path):
    token_file = tmp_path / DEFAULT_TOKEN_FILE
    token_file.write_text("file-token\n", encoding="utf-8")
    monkeypatch.setenv("IBM_API_KEY", "env-token")
    monkeypatch.chdir(tmp_path)
    token = load_ibm_token()
    assert token == "env-token"


def test_token_file_disabled(monkeypatch, tmp_path):
    token_file = tmp_path / DEFAULT_TOKEN_FILE
    token_file.write_text("file-token\n", encoding="utf-8")
    monkeypatch.delenv("IBM_API_KEY", raising=False)
    monkeypatch.delenv("IBM_API_KEY_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Missing IBM Quantum token"):
        load_ibm_token(token_file=None)


def test_missing_token_raises_with_guidance(monkeypatch, tmp_path):
    monkeypatch.delenv("IBM_API_KEY", raising=False)
    monkeypatch.delenv("IBM_API_KEY_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        load_ibm_token()
    msg = str(exc.value)
    assert "IBM_API_KEY" in msg
    assert "IBM_API_KEY_FILE" in msg
    assert DEFAULT_TOKEN_FILE in msg


def test_custom_api_key_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_IBM_TOKEN", "custom-env-token")
    monkeypatch.delenv("IBM_API_KEY", raising=False)
    monkeypatch.delenv("IBM_API_KEY_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    token = load_ibm_token(api_key_env="MY_IBM_TOKEN", token_file=None)
    assert token == "custom-env-token"


def test_token_file_absolute_path(monkeypatch, tmp_path):
    token_file = tmp_path / "abs_token"
    token_file.write_text("abs-token\n", encoding="utf-8")
    monkeypatch.delenv("IBM_API_KEY", raising=False)
    monkeypatch.delenv("IBM_API_KEY_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    token = load_ibm_token(token_file=token_file)
    assert token == "abs-token"

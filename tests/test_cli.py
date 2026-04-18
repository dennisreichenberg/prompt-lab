"""Tests for prompt-lab CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from prompt_lab.cli import app
from prompt_lab.ollama import ModelResult
from prompt_lab.templates import all_templates, delete_template, save_template

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    store = tmp_path / "templates.json"
    monkeypatch.setattr("prompt_lab.templates._STORE", store)
    return store


def _mock_result(model: str, response: str = "test response") -> ModelResult:
    return ModelResult(model=model, response=response, elapsed_seconds=0.5)


def test_list_empty():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No templates" in result.output


def test_save_and_list():
    result = runner.invoke(app, ["save", "mytest", "Tell me a joke"])
    assert result.exit_code == 0
    assert "mytest" in result.output

    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "mytest" in result.output
    assert "Tell me a joke" in result.output


def test_delete_template():
    save_template("todel", "some prompt")
    result = runner.invoke(app, ["delete", "todel"])
    assert result.exit_code == 0
    assert "todel" in result.output
    assert all_templates() == {}


def test_delete_nonexistent():
    result = runner.invoke(app, ["delete", "ghost"])
    assert result.exit_code != 0


def test_run_no_prompt():
    result = runner.invoke(app, ["run", "--models", "mistral"])
    assert result.exit_code != 0


@patch("prompt_lab.cli._client")
def test_run_with_prompt(mock_client):
    client = MagicMock()
    client.generate.return_value = _mock_result("mistral")
    mock_client.return_value = client

    result = runner.invoke(app, ["run", "Hello world", "--models", "mistral"])
    assert result.exit_code == 0
    assert "mistral" in result.output


@patch("prompt_lab.cli._client")
def test_run_all_models(mock_client):
    client = MagicMock()
    client.list_model_names.return_value = ["mistral", "phi3"]
    client.generate.side_effect = lambda m, p: _mock_result(m)
    mock_client.return_value = client

    result = runner.invoke(app, ["run", "Hello", "--models", "all"])
    assert result.exit_code == 0
    assert "mistral" in result.output
    assert "phi3" in result.output


@patch("prompt_lab.cli._client")
def test_run_from_file(mock_client, tmp_path):
    client = MagicMock()
    client.generate.return_value = _mock_result("llama3")
    mock_client.return_value = client

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("What is 2+2?")

    result = runner.invoke(app, ["run", "--models", "llama3", "--file", str(prompt_file)])
    assert result.exit_code == 0
    assert "llama3" in result.output


@patch("prompt_lab.cli._client")
def test_run_from_template(mock_client):
    save_template("myq", "What is the capital of France?")

    client = MagicMock()
    client.generate.return_value = _mock_result("mistral", "Paris")
    mock_client.return_value = client

    result = runner.invoke(app, ["run", "--models", "mistral", "--template", "myq"])
    assert result.exit_code == 0
    assert "mistral" in result.output


@patch("prompt_lab.cli._client")
def test_run_error_model(mock_client):
    client = MagicMock()
    client.generate.return_value = ModelResult(
        model="broken", response="", elapsed_seconds=0.1, error="connection refused"
    )
    mock_client.return_value = client

    result = runner.invoke(app, ["run", "hi", "--models", "broken"])
    assert result.exit_code == 0
    assert "broken" in result.output

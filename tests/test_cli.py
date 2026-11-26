from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from reward_model.cli import main


@pytest.fixture(scope="module")
def workspace(raw_file: tuple[Path, str], tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("cli")
    path, digest = raw_file
    args = ["--data-dir", str(root / "data"), "--artifacts", str(root / "artifacts")]
    assert main(["download", *args, "--url", path.as_uri(), "--sha256", digest]) == 0
    assert main(["baselines", *args, "--train-limit", "150"]) == 0
    return root


def _args(root: Path) -> list[str]:
    return ["--data-dir", str(root / "data"), "--artifacts", str(root / "artifacts")]


def test_download_and_baselines_write_files(workspace: Path) -> None:
    assert (workspace / "data" / "pairs" / "train.jsonl").exists()
    assert (workspace / "artifacts" / "baselines.joblib").exists()
    scores = sorted(p.stem for p in (workspace / "artifacts" / "scores").glob("*.json"))
    assert scores == ["bt_hand", "bt_tfidf", "length", "random"]
    meta = json.loads((workspace / "artifacts" / "scores" / "bt_tfidf.json").read_text())["meta"]
    assert meta["train_pairs"] == 150
    assert meta["command"].startswith("reward-model baselines")


def test_report_command(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    reports = workspace / "reports"
    assert main(["report", *_args(workspace), "--reports", str(reports), "--no-charts"]) == 0
    report = json.loads((reports / "metrics.json").read_text())
    assert report["data"]["pairs"]["test"] == report["test_set"]["pairs"]
    assert "bt_tfidf" in capsys.readouterr().out


def test_score_command_with_baseline(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "score",
            *_args(workspace),
            "--baseline",
            "bt_tfidf",
            "--prompt",
            "Give me practical tips on sourdough bread.",
            "--response",
            "1. Check the starter first, then adjust the flour carefully.\n"
            "2. Check the oven first, then adjust the crust carefully.",
            "--response",
            "It is basically about stuff things.",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"] == "bt_tfidf"
    assert out["rewards"][0] > out["rewards"][1]
    assert out["p_first_beats_second"] > 0.5


def test_report_without_scores_fails_cleanly(tmp_path: Path) -> None:
    assert main(["report", "--data-dir", str(tmp_path), "--artifacts", str(tmp_path)]) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="torch is installed")
def test_train_without_extra_explains_how_to_install(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["train", *_args(workspace)]) == 2
    assert "uv sync --extra train" in capsys.readouterr().err

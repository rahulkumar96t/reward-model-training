from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import make_rows, write_raw

from reward_model.data import PreferencePair, load_pairs, prepare


@pytest.fixture(scope="session")
def raw_file(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    path = tmp_path_factory.mktemp("raw") / "preference.jsonl.gz"
    digest = write_raw(path, make_rows(n=400, seed=1))
    return path, digest


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory, raw_file: tuple[Path, str]) -> Path:
    folder = tmp_path_factory.mktemp("data")
    prepare(raw_file[0], folder)
    return folder


@pytest.fixture(scope="session")
def splits(data_dir: Path) -> dict[str, list[PreferencePair]]:
    return {split: load_pairs(data_dir, split) for split in ("train", "val", "test")}

.PHONY: install install-train data baselines train report test lint clean

install:
	uv sync --locked

# PyTorch + transformers for the neural reward model (about 1 GB)
install-train:
	uv sync --locked --extra train

# Downloads preference.jsonl.gz (15 MB, SHA-256 checked) and writes data/pairs/{train,val,test}.jsonl
data:
	uv run reward-model download

baselines:
	uv run reward-model baselines

# The measured run: full train split, 1 epoch, MPS on Apple Silicon, about 18 min (see README)
train:
	uv run --extra train reward-model train --epochs 1 --evals-per-epoch 3 --grad-checkpointing

report:
	uv run reward-model report

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

clean:
	rm -rf artifacts .pytest_cache .ruff_cache

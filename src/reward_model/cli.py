"""``reward-model`` command line: download, baselines, train, report, score."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .data import (
    LICENSE,
    REVISION,
    SHA256,
    URL,
    download,
    load_pairs,
    load_summary,
    prepare,
    sample_pairs,
)
from .metrics import fit_temperature
from .scoring import collect_scores, load_score_sets, scores_path

DEFAULT_DATA = Path("data")
DEFAULT_ARTIFACTS = Path("artifacts")
DEFAULT_REPORTS = Path("reports")


def _command(args: argparse.Namespace) -> str:
    return " ".join(["reward-model", *args.argv])


def _machine() -> dict[str, str]:
    return {"platform": platform.platform(), "python": platform.python_version()}


def cmd_download(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    raw = download(args.data_dir, url=args.url, sha256=args.sha256)
    print(f"verified {raw} (sha256 {args.sha256[:12]}...)")
    summary = prepare(raw, args.data_dir)
    for split in ("train", "val", "test"):
        print(f"  {split:<5} {summary.pairs[split]:>5} pairs, {summary.prompts[split]:>5} prompts")
    print(
        f"dropped: {summary.dropped_ties} ties, {summary.dropped_identical} identical, "
        f"{summary.dropped_empty} empty, {summary.dropped_duplicates} duplicates "
        f"(of {summary.raw_rows} rows); {time.perf_counter() - start:.1f}s"
    )
    print(f"licence: {LICENSE}; source revision {REVISION}")
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    from .baselines import save_baselines, train_baselines

    train = load_pairs(args.data_dir, "train")
    val = load_pairs(args.data_dir, "val")
    test = load_pairs(args.data_dir, "test")
    if args.train_limit:
        train = sample_pairs(train, args.train_limit, args.seed)
    print(f"fitting baselines on {len(train)} train pairs (val {len(val)}, test {len(test)})")
    start = time.perf_counter()
    scorers, info = train_baselines(train, val, seed=args.seed)
    fit_seconds = time.perf_counter() - start
    temperatures: dict[str, float] = {}
    for name, scorer in scorers.items():
        t0 = time.perf_counter()
        meta = {
            "command": _command(args),
            "train_pairs": len(train),
            "fit_seconds_all_baselines": round(fit_seconds, 2),
            **info[name],
            **_machine(),
        }
        scores = collect_scores(scorer, val, test, seed=args.seed, meta=meta)
        scores.meta["score_seconds"] = round(time.perf_counter() - t0, 2)
        temperatures[name] = fit_temperature(scores.val.margins)
        scores.meta["temperature"] = round(temperatures[name], 4)
        path = scores.save(scores_path(args.artifacts, name))
        print(f"  {name:<9} val acc {np.mean(scores.val.margins > 0):.4f} -> {path}")
    save_baselines(
        {"scorers": scorers, "temperatures": temperatures}, args.artifacts / "baselines.joblib"
    )
    print(f"done in {time.perf_counter() - start:.1f}s")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    try:
        import torch
        import transformers

        from . import neural
    except ImportError:
        print("the neural model needs the train extra: uv sync --extra train", file=sys.stderr)
        return 2

    torch.set_num_threads(args.threads)
    config = neural.TrainConfig(
        model_name=args.model_name,
        revision=args.revision or None,
        max_length=args.max_length,
        max_prompt_tokens=args.max_prompt_tokens,
        pooling=args.pooling,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin_scale=args.margin_scale,
        center_coef=args.center_coef,
        grad_checkpointing=args.grad_checkpointing,
        evals_per_epoch=args.evals_per_epoch,
        seed=args.seed,
        device=args.device,
        threads=args.threads,
    )
    train = load_pairs(args.data_dir, "train")
    val = load_pairs(args.data_dir, "val")
    test = load_pairs(args.data_dir, "test")
    if args.train_limit:
        train = sample_pairs(train, args.train_limit, args.seed)
    if args.eval_limit:
        val = sample_pairs(val, args.eval_limit, args.seed)
        test = sample_pairs(test, args.eval_limit, args.seed)
    device = neural.pick_device(config.device)
    print(f"training {config.model_name} on {len(train)} pairs, device {device}")
    wall, clock = time.perf_counter(), time.time()
    model, tokenizer = neural.build_model(config)
    encoder = neural.PairEncoder(tokenizer, config.budget)
    result = neural.train_reward_model(model, encoder, config, train, val, device)
    out_dir = args.artifacts / args.name
    extra = {"temperature": result.temperature, "best": result.best}
    neural.save_checkpoint(model, tokenizer, config, out_dir, extra)
    print(f"saved {out_dir} (best {result.best})")

    scorer = neural.NeuralScorer(
        model, encoder, device, config.eval_batch_size, result.temperature, name=args.name
    )
    t0 = time.perf_counter()
    meta: dict[str, Any] = {
        "command": _command(args),
        "train_pairs": len(train),
        "device": str(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "config": dict(vars(config)),
        "steps": result.steps,
        "train_seconds": result.train_seconds,
        "history": result.history,
        "best": result.best,
        "temperature": round(result.temperature, 4),
        "response_truncated": result.truncated,
        "peak_mps_driver_gb": result.peak_mps_gb,
        **_machine(),
    }
    scores = collect_scores(scorer, val, test, seed=args.seed, meta=meta)
    _, test_cut = encoder.encode(
        [p.prompt for p in test] * 2, [p.chosen for p in test] + [p.rejected for p in test]
    )
    scores.meta["response_truncated"]["test"] = round(test_cut, 4)
    scores.meta["score_seconds"] = round(time.perf_counter() - t0, 1)
    # perf_counter stops while the machine sleeps; the wall clock does not.
    scores.meta["wall_seconds"] = round(time.perf_counter() - wall, 1)
    scores.meta["wall_clock_seconds"] = round(time.time() - clock, 1)
    path = scores.save(scores_path(args.artifacts, args.name))
    print(f"test acc {np.mean(scores.test.margins > 0):.4f}; scores -> {path}")
    print(
        f"time {scores.meta['wall_seconds']:.0f}s (wall clock "
        f"{scores.meta['wall_clock_seconds']:.0f}s)"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report, summary_table, write_report

    score_sets = load_score_sets(args.artifacts)
    if not score_sets:
        print("no score files; run `reward-model baselines` first", file=sys.stderr)
        return 1
    test = load_pairs(args.data_dir, "test")
    report = build_report(score_sets, test, load_summary(args.data_dir), seed=args.seed)
    report["version"] = __version__
    for path in write_report(report, args.reports, charts=not args.no_charts):
        print(f"wrote {path}")
    print(summary_table(report))
    return 0


def _load_scorer(args: argparse.Namespace) -> tuple[Any, float]:
    if args.checkpoint:
        from .neural import load_checkpoint

        scorer = load_checkpoint(args.checkpoint, device=args.device)
        return scorer, scorer.temperature
    from .baselines import load_baselines

    bundle = load_baselines(args.artifacts / "baselines.joblib")
    if args.baseline not in bundle["scorers"]:
        raise SystemExit(f"unknown baseline {args.baseline!r}: {sorted(bundle['scorers'])}")
    return bundle["scorers"][args.baseline], bundle["temperatures"][args.baseline]


def cmd_score(args: argparse.Namespace) -> int:
    scorer, temperature = _load_scorer(args)
    rewards = scorer.score([args.prompt] * len(args.response), args.response)
    out: dict[str, Any] = {
        "model": getattr(scorer, "name", "?"),
        "rewards": [round(float(r), 4) for r in rewards],
    }
    if len(rewards) >= 2:
        margin = float(rewards[0] - rewards[1])
        out["p_first_beats_second"] = round(float(1 / (1 + np.exp(-margin / temperature))), 4)
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reward-model", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
        p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
        p.add_argument("--seed", type=int, default=13)

    p = sub.add_parser("download", help="download, verify and split HelpSteer2 preferences")
    common(p)
    p.add_argument("--url", default=URL)
    p.add_argument("--sha256", default=SHA256)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("baselines", help="fit random / length / Bradley-Terry baselines")
    common(p)
    p.add_argument("--train-limit", type=int, default=0, help="use a seeded subset")
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("train", help="train the transformer reward model (train extra)")
    common(p)
    p.add_argument("--name", default="neural", help="checkpoint folder and score-file name")
    p.add_argument("--model-name", default="distilbert/distilroberta-base")
    p.add_argument("--revision", default="fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b")
    p.add_argument("--train-limit", type=int, default=0, help="use a seeded subset")
    p.add_argument(
        "--eval-limit", type=int, default=0, help="subset val and test too (smoke runs only)"
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8, help="pairs per step")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=128)
    p.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    p.add_argument("--margin-scale", type=float, default=0.0)
    p.add_argument("--center-coef", type=float, default=0.0)
    p.add_argument(
        "--grad-checkpointing", action="store_true", help="less activation memory, slower steps"
    )
    p.add_argument("--evals-per-epoch", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("report", help="write reports/metrics.json and charts from score files")
    common(p)
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--no-charts", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("score", help="score responses to one prompt")
    common(p)
    p.add_argument("--prompt", required=True)
    p.add_argument("--response", action="append", required=True, help="repeat for several")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="neural checkpoint folder")
    source.add_argument("--baseline", default="bt_tfidf", help="baseline name")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    args.argv = list(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

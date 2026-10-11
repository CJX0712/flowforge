"""Command-line entry point for FlowForge.

    python -m flowforge.cli benchmark --datasets warp6 banana6 --seeds 1 2 3
    python -m flowforge.cli fit --dataset moons --model realnvp
    python -m flowforge.cli selftest

Author: 晨星
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .core.config import Config
from .core.seed import set_all
from .data.preprocess import standardize_dataset
from .data.synthetic import discriminative_datasets, list_datasets
from .eval.metrics import negative_log_likelihood
from .flows.maf import MAFStack
from .flows.realnvp import RealNVP
from .pipeline.pipeline import FlowForgePipeline
from .training.trainer import fit


def _print_table(rep) -> None:
    header = f"{'dataset':<12}{'model':<14}{'kind':<10}{'NLL mean':>12}{'±std':>10}{'params':>10}"
    print(header)
    print("-" * len(header))
    for r in sorted(rep.rows, key=lambda x: (x.dataset, x.kind != "flagship", x.nll_mean)):
        print(f"{r.dataset:<12}{r.model:<14}{r.kind:<10}{r.nll_mean:>12.4f}{r.nll_std:>10.4f}{r.n_params:>10}")


def cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = Config(
        seed=args.seed,
        epochs=args.epochs,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        lr=args.lr,
        hidden=args.hidden,
    )
    set_all(args.seed)
    pip = FlowForgePipeline(cfg)
    t0 = time.time()
    rep = pip.benchmark(args.datasets, args.seeds, verbose=args.verbose)
    rep.meta["elapsed_sec"] = round(time.time() - t0, 3)
    _print_table(rep)
    print(f"\nelapsed: {rep.meta['elapsed_sec']}s")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "meta": rep.meta,
                    "rows": [
                        {
                            "dataset": r.dataset,
                            "model": r.model,
                            "kind": r.kind,
                            "nll_mean": r.nll_mean,
                            "nll_std": r.nll_std,
                            "n_params": r.n_params,
                        }
                        for r in rep.rows
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    set_all(args.seed)
    cfg = Config(seed=args.seed, epochs=args.epochs, lr=args.lr)
    from .data.synthetic import make_dataset

    ds = make_dataset(args.dataset, n_train=cfg.n_train, n_val=cfg.n_val, n_test=cfg.n_test, seed=cfg.seed)
    ds, _ = standardize_dataset(ds)
    model = (
        RealNVP(ds.dim, n_blocks=6, hidden=cfg.hidden, seed=cfg.seed)
        if args.model == "realnvp"
        else MAFStack(ds.dim, n_blocks=2, hidden=cfg.hidden, seed=cfg.seed)
    )
    res = fit(
        model,
        ds.X,
        ds.X_val,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        patience=cfg.patience,
        seed=cfg.seed,
    )
    print(
        f"model={args.model} dataset={args.dataset} "
        f"train_nll {res.train_nll_first:.4f} -> {res.train_nll_last:.4f} | "
        f"best_val_nll={res.best_val_nll:.4f} | "
        f"test_nll={negative_log_likelihood(model, ds.X_test):.4f} | epochs={res.epochs_run}"
    )
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Fast invariant self-check (no pytest required)."""
    import numpy as np

    from .core.seed import spawn
    from .flows.base import base_logprob

    set_all(0)
    ok = True
    model = RealNVP(dim=3, n_blocks=3, hidden=16, seed=7)
    x = spawn(0).normal(size=(16, 3))
    z, _ = model.forward(x)
    err = float(np.max(np.abs(model.inverse(z) - x)))
    print(f"[{'OK' if err < 1e-10 else 'FAIL'}] invertibility err={err:.3e}")
    ok &= err < 1e-10

    z2, ld = model.forward(x)
    n = x.shape[0]
    _, grads = model.backward(z2 / n, -np.ones(n) / n)

    def loss():
        zz, ll = model.forward(x)
        return float(-np.mean(base_logprob(zz) + ll))

    from .training.trainer import collect_pointers

    pointers = collect_pointers(model)
    worst = 0.0
    for key, (container, sub) in list(pointers.items())[:5]:
        base = container[sub].copy()
        idx = np.unravel_index(np.argmin(np.abs(base)) + 1, base.shape)
        eps = 1e-6
        container[sub][idx] = base[idx] + eps
        lp = loss()
        container[sub][idx] = base[idx] - eps
        lm = loss()
        container[sub][idx] = base[idx]
        num = (lp - lm) / (2 * eps)
        worst = max(worst, abs(num - float(grads[key][idx])) / max(1.0, abs(num)))
    print(f"[{'OK' if worst < 1e-6 else 'FAIL'}] gradient check worst_rel={worst:.3e}")
    ok &= worst < 1e-6
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flowforge", description="FlowForge CLI (author: 晨星)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="run the density-estimation benchmark")
    b.add_argument("--datasets", nargs="+", default=discriminative_datasets())
    b.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    b.add_argument("--epochs", type=int, default=250)
    b.add_argument("--n-train", type=int, default=2000)
    b.add_argument("--n-val", type=int, default=800)
    b.add_argument("--n-test", type=int, default=1500)
    b.add_argument("--lr", type=float, default=8e-3)
    b.add_argument("--hidden", type=int, default=64)
    b.add_argument("--seed", type=int, default=12345)
    b.add_argument("--out", default="benchmark.json")
    b.add_argument("--verbose", action="store_true")
    b.set_defaults(func=cmd_benchmark)

    f = sub.add_parser("fit", help="fit a single flow")
    f.add_argument("--dataset", default="warp6", choices=list_datasets())
    f.add_argument("--model", default="realnvp", choices=["realnvp", "maf"])
    f.add_argument("--epochs", type=int, default=250)
    f.add_argument("--lr", type=float, default=8e-3)
    f.add_argument("--seed", type=int, default=12345)
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("selftest", help="fast invariant self-check")
    s.set_defaults(func=cmd_selftest)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

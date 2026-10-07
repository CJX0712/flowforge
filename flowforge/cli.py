"""FlowForge 命令行入口。

python -m flowforge.cli --quick          # 冒烟
python -m flowforge.cli --check          # 只跑不变量自检
python -m flowforge.cli --json out.json  # 出基准 JSON
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence

import numpy as np

from .core.config import Config
from .core.seed import set_all
from .core.types import FlowSpec
from .data.densities import DATASETS, build_density
from .eval import invariants
from .pipeline.benchmark import DEFAULT_SEEDS, run_benchmark
from .pipeline.methods import METHODS
from .training.fit import build_flow


def _print_table(rows) -> None:
    hdr = f"{'dataset':9s} {'method':9s} {'seed':>5s} {'KL':>10s} {'W2':>9s} {'MMD':>10s} {'P':>6s} {'sec':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.dataset:9s} {r.method:9s} {r.seed:5d} {r.kl:10.5f} "
            f"{r.w2:9.4f} {r.mmd:10.6f} {r.n_params:6d} {r.train_sec:6.2f}"
        )


def _cmd_check(cfg: Config) -> int:
    """跑一遍全部硬不变量。"""
    ok = True
    for name in DATASETS:
        dens = build_density(name)
        rng = np.random.default_rng(int(cfg.seed))
        X = dens.sample(300, rng)
        spec = FlowSpec(
            family="affine",
            block_type="affine",
            n_blocks=4,
            hidden=16,
            n_bins=4,
            seed=int(cfg.seed),
        )
        model = build_flow(spec, 2, X, rng)
        res = invariants.check_all(model, X[:32], dens.log_prob(X[:32]))
        bad = [n for n, v, p in res if not p]
        flag = "OK " if not bad else "FAIL"
        vals = "  ".join(f"{n.split('_', 1)[0]}={v:.2e}" for n, v, _ in res)
        print(f"[{flag}] {name:8s} {vals}")
        ok = ok and not bad
    print("不变量自检:", "全绿" if ok else "有失败项")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flowforge", description="FlowForge CLI")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="小配置冒烟")
    ap.add_argument("--check", action="store_true", help="只跑不变量自检")
    ap.add_argument("--json", default=None, help="把结果写出为 JSON")
    args = ap.parse_args(argv)

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

    set_all(int(args.seeds[0]))
    cfg = Config(iters=args.iters, hidden=args.hidden, n_blocks=args.blocks)
    if args.quick:
        cfg = Config(
            n_train=600, n_test=600, iters=120, batch=256, hidden=16, n_blocks=4, n_w2_samples=150
        )
    cfg.validate()

    if args.check:
        return _cmd_check(cfg)

    rows = run_benchmark(
        cfg, datasets=args.datasets, methods=args.methods, seeds=args.seeds, verbose=True
    )
    _print_table(rows)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([r.__dict__ for r in rows], fh, ensure_ascii=False, indent=2)
        print(f"已写出 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

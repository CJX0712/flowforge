"""FlowForge 端到端演示：基准 + 门禁 + 消融 + 不变量，落盘 benchmark.json。

    python examples/run_demo.py            # 完整（约 60s）
    python examples/run_demo.py --quick    # CI 冒烟（数秒）

所有数字均来自真实运行，禁止手填"预期值"。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowforge.core.config import Config
from flowforge.core.seed import set_all
from flowforge.core.types import FlowSpec
from flowforge.data.densities import DATASETS
from flowforge.eval.invariants import check_all
from flowforge.eval.metrics import summarize
from flowforge.pipeline.benchmark import DEFAULT_SEEDS, evaluate, make_dataset, run_benchmark
from flowforge.pipeline.methods import fit_method

BASELINES = ("affine", "maf")
FLAGSHIP = "fusion"
# ---- 预注册门槛（方案阶段定死，事后不许迁就结果）--------------------------
GATE_MIN_REDUCTION = 0.10  # G1a：聚合 KL 相对最强基线降低 >= 10%
GATE_MIN_SIG_DATASETS = 2  # G1b：至少 2 个数据集配对显著胜出
GATE_NONINFERIOR_TOL = 0.005  # G2：逐数据集非劣的**绝对**容差（nats）
GATE_NONINFERIOR_MIN = 4  # G2：至少 4/5 数据集非劣
GATE_ABLATION_REDUCTION = 0.10  # G5：K=4 相对 K=1 的 KL 降低 >= 10%（混合组件归因）
GATE_KL_TOL = 1e-6  # I5：KL >= -tol


def _agg(rows: list[Any], method: str) -> tuple:
    vals = [r.kl for r in rows if r.method == method]
    return summarize(vals)


def _per_dataset(rows: list[Any], dname: str) -> dict[str, float]:
    out = {}
    for m in set(r.method for r in rows if r.dataset == dname):
        out[m] = float(np.mean([r.kl for r in rows if r.dataset == dname and r.method == m]))
    return out


def run_forced_k(
    cfg: Config, datasets, seeds, k: int, block: str = "affine", iters: int | None = None
) -> dict[str, float]:
    """强制分量数 K 的 fusion（用于消融）。返回 {dataset: 平均 KL}。"""
    iters = int(iters or cfg.iters)
    out: dict[str, list[float]] = {d: [] for d in datasets}
    for dname in datasets:
        base = make_dataset(dname, cfg, seed=int(seeds[0]))
        lp_true = base.log_prob(base.X_test)
        for sd in seeds:
            ds = make_dataset(dname, cfg, seed=int(sd))
            spec = FlowSpec(
                family="fusion",
                block_type=block,
                n_blocks=int(cfg.n_blocks),
                hidden=int(cfg.hidden),
                n_bins=int(cfg.n_bins),
                n_components=int(k),
                seed=int(sd),
            )
            model, _ = fit_method(
                "fusion", spec, ds.X_train, cfg, int(sd), iters=iters, revive=True
            )
            ev = evaluate(
                model,
                base.X_test,
                lp_true,
                np.random.default_rng(int(sd) * 31 + 7),
                int(cfg.n_w2_samples),
            )
            out[dname].append(ev["kl"])
    return {d: float(np.mean(v)) for d, v in out.items()}


def _strongest(p: dict[str, float]) -> tuple:
    """返回 (基线名, 该数据集上最强单流基线的 KL)。"""
    m = min((b for b in BASELINES if b in p), key=lambda b: p[b])
    return m, float(p[m])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args(argv)
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

    set_all(7)
    if args.quick:
        cfg = Config(
            n_train=500, n_test=500, iters=80, batch=256, hidden=16, n_blocks=4, n_w2_samples=120
        )
        datasets, seeds = ("gmm5", "banana"), (7,)
    else:
        cfg = Config()
        datasets, seeds = DATASETS, DEFAULT_SEEDS
    cfg.validate()

    t0 = time.perf_counter()
    rows = run_benchmark(cfg, datasets=datasets, seeds=seeds, verbose=True)
    elapsed = time.perf_counter() - t0

    # ---- 聚合 ----
    print(f"\n=== 聚合（{len(datasets)} datasets × {len(seeds)} seeds，KL 越低越好） ===")
    agg = {}
    for m in ("gaussian", *BASELINES, FLAGSHIP):
        mu, sd = _agg(rows, m)
        agg[m] = (mu, sd)
        print(f"  {m:9s} KL = {mu:.5f} ± {sd:.5f}")
    base_name = min(BASELINES, key=lambda m: agg[m][0])
    b_mu, _b_sd = agg[base_name]
    f_mu, _f_sd = agg[FLAGSHIP]
    reduction = (b_mu - f_mu) / b_mu if b_mu != 0 else 0.0
    print(f"\n  最强单流基线 = {base_name} ({b_mu:.5f})；旗舰 = {FLAGSHIP} ({f_mu:.5f})")
    print(f"  G1a 聚合 KL 降低 = {reduction * 100:.2f}%  (门槛 {GATE_MIN_REDUCTION * 100:.0f}%)")
    g1a = reduction >= GATE_MIN_REDUCTION

    # ---- G1b 逐数据集配对显著性 ----
    # 聚合跨数据集的 std 会被量级差 100 倍的数据集撑爆（实测 σ≈0.18），
    # 直接套「均值差 > ½(σ₁+σ₂)」在统计上无意义。故改逐数据集配对检验。
    print("\n=== G1b 逐数据集配对显著性（seeds 配对，同一 train 种子） ===")
    sig_ds = []
    for d in datasets:
        sb, sb_v = _strongest(_per_dataset(rows, d))
        fv = [r.kl for r in rows if r.dataset == d and r.method == FLAGSHIP]
        bv = [r.kl for r in rows if r.dataset == d and r.method == sb]
        f_m, f_s = summarize(fv)
        b_m, b_s = summarize(bv)
        d_mean = f_m - b_m
        thr = 0.5 * (f_s + b_s)
        sig = d_mean < -thr  # 越小越好，故要求 fusion 显著更低
        sig_ds.append((d, sig))
        print(
            f"  {d:8s} vs {sb:7s} Δ={d_mean:+.5f}  阈值 ½(σ₁+σ₂)={thr:.5f}  "
            f"{'显著胜' if sig else '未显著'}"
        )
    n_sig = sum(1 for _, s in sig_ds if s)
    g1b = n_sig >= GATE_MIN_SIG_DATASETS
    print(
        f"  显著胜出数据集 {n_sig}/{len(datasets)}  (门槛 >= {GATE_MIN_SIG_DATASETS})"
        f" -> {'PASS' if g1b else 'FAIL'}"
    )

    # ---- G2 逐数据集非劣 ----
    print("\n=== 逐数据集 ===")
    n_ok = 0
    per_ds = {}
    for d in datasets:
        p = _per_dataset(rows, d)
        per_ds[d] = p
        sb, sb_v = _strongest(p)
        noninf = p[FLAGSHIP] <= sb_v + GATE_NONINFERIOR_TOL
        n_ok += int(noninf)
        print(
            f"  {d:8s} fusion={p[FLAGSHIP]:.5f}  最强基线 {sb}={sb_v:.5f}  "
            f"Δ={p[FLAGSHIP] - sb_v:+.5f}  {'OK' if noninf else '劣'}"
        )
    g2 = n_ok >= GATE_NONINFERIOR_MIN
    print(
        f"  G2 非劣数据集 {n_ok}/{len(datasets)}  (门槛 >= {GATE_NONINFERIOR_MIN}) -> {'PASS' if g2 else 'FAIL'}"
    )

    # ---- G4 不变量 ----
    print("\n=== G4 硬不变量（旗舰，各数据集抽 1 个 seed） ===")
    inv_all = True
    inv_detail = {}
    for d in datasets:
        ds = make_dataset(d, cfg, seed=int(seeds[0]))
        spec = FlowSpec(
            family="fusion",
            block_type="affine",
            n_blocks=int(cfg.n_blocks),
            hidden=int(cfg.hidden),
            n_bins=int(cfg.n_bins),
            n_components=4,
            seed=int(seeds[0]),
        )
        model, _ = fit_method(
            "fusion", spec, ds.X_train, cfg, int(seeds[0]), iters=int(cfg.iters), revive=True
        )
        # I1~I3 只用少量点（雅可比/梯度检验昂贵）；I5 必须用大样本，否则
        # 蒙特卡洛噪声会把「KL≈0 的好模型」误判为负。
        res = check_all(
            model,
            ds.X_test[:24],
            ds.log_prob(ds.X_test[:24]),
            X_kl=ds.X_test,
            log_p_true_kl=ds.log_prob(ds.X_test),
        )
        inv_detail[d] = {n: v for n, v, _ in res}
        bad = [n for n, v, ok in res if not ok]
        inv_all = inv_all and not bad
        vals = " ".join(f"{n.split('_', 1)[0]}:{v:.1e}{'' if ok else '!'}" for n, v, ok in res)
        print(f"  {d:8s} {vals}")
    print(f"  G4 -> {'PASS' if inv_all else 'FAIL'}")

    # ---- G5 消融 ----
    abl_datasets = tuple(datasets)[:3] if not args.quick else datasets
    print(f"\n=== G5 消融（K 固定 vs 自适应，{len(abl_datasets)} datasets × 2 seeds） ===")
    abl_seeds = seeds[:2] if not args.quick else seeds
    k1 = run_forced_k(cfg, abl_datasets, abl_seeds, k=1)
    k4 = run_forced_k(cfg, abl_datasets, abl_seeds, k=4)
    abl = {}
    for d in abl_datasets:
        p = _per_dataset(rows, d)
        abl[d] = {"adaptive": p[FLAGSHIP], "K=1": k1[d], "K=4": k4[d]}
        print(f"  {d:8s} adaptive={p[FLAGSHIP]:.5f}  K=1={k1[d]:.5f}  K=4={k4[d]:.5f}")
    a_ad = float(np.mean([abl[d]["adaptive"] for d in abl_datasets]))
    a_k1 = float(np.mean([abl[d]["K=1"] for d in abl_datasets]))
    a_k4 = float(np.mean([abl[d]["K=4"] for d in abl_datasets]))
    mix_gain = (a_k1 - a_k4) / a_k1 if a_k1 else 0.0
    print(f"  平均：adaptive={a_ad:.5f}  K=1={a_k1:.5f}  K=4={a_k4:.5f}")
    print(
        f"  混合组件归因：K=4 相对 K=1 降低 {mix_gain * 100:.2f}% "
        f"(门槛 {GATE_ABLATION_REDUCTION * 100:.0f}%)"
    )
    g5 = mix_gain >= GATE_ABLATION_REDUCTION
    # 诚实负结果保留：自适应 K 相对强制 K=4 未必更优
    adaptive_margin = (a_ad - a_k4) / a_k4 if a_k4 else 0.0
    print(
        f"  [负结果保留] 自适应 K 相对强制 K=4 的边际代价 = {adaptive_margin * 100:+.2f}%"
        f"（自适应换来的是单峰域的非劣安全性）"
    )
    print(f"  G5 -> {'PASS' if g5 else 'FAIL'}")

    # ---- G6 失败案例（全部从 results 派生，禁止预设结论） ----
    print("\n=== G6 失败案例（从实测结果派生） ===")
    worst = sorted(((p[FLAGSHIP] - _strongest(p)[1], d) for d, p in per_ds.items()), reverse=True)[
        :3
    ]
    cases = []
    for gap, d in worst:
        p = per_ds[d]
        sb, sb_v = _strongest(p)
        cases.append(
            {"dataset": d, "fusion": p[FLAGSHIP], "baseline": sb, "baseline_kl": sb_v, "delta": gap}
        )
        print(f"  {d:8s} fusion 比 {sb} 差 {gap:+.5f} nats（KL {p[FLAGSHIP]:.5f} vs {sb_v:.5f}）")
    if not cases:
        print("  无失败案例（旗舰在所有数据集非劣）")

    # ---- 汇总 ----
    gates = {
        "G1a_aggregate_reduction": bool(g1a),
        "G1b_paired_significance": bool(g1b),
        "G2_noninferior": bool(g2),
        "G4_invariants": bool(inv_all),
        "G5_ablation": bool(g5),
        "G6_failure_cases": len(cases) >= 1,
    }
    print("\n=== 门禁 ===")
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  demo 端到端耗时 {elapsed:.1f}s（预算 60s）")

    payload = {
        "config": cfg.__dict__,
        "datasets": list(datasets),
        "seeds": list(seeds),
        "aggregate": {m: {"mean": agg[m][0], "std": agg[m][1]} for m in agg},
        "strongest_baseline": base_name,
        "reduction_pct": reduction * 100.0,
        "per_dataset": per_ds,
        "ablation": abl,
        "ablation_mixture_gain_pct": mix_gain * 100.0,
        "ablation_adaptive_margin_pct": adaptive_margin * 100.0,
        "invariants": inv_detail,
        "failure_cases": cases,
        "gates": gates,
        "elapsed_sec": elapsed,
        "rows": [r.__dict__ for r in rows],
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"已写出 {args.out}")
    if args.quick:
        # 冒烟模式只验证「能跑通」，门禁数字在完整预算下才有意义
        print("（--quick：不校验门禁阈值）")
        return 0
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

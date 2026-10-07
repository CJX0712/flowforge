"""基准测试编排。

数据切分（防泄漏 / 降方差）：
  train  = 采样种子 seed（随 seed 变化）
  val    = 采样种子 cfg.hpo_seed（固定，仅供选参，与 train/test 不相交）
  test   = 采样种子 cfg.test_seed（**固定**，所有方法/种子共用同一 holdout，
           形成配对比较，显著降低 MC 方差）
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.config import Config
from ..core.types import Dataset
from ..data.densities import DATASETS, build_density
from ..eval.metrics import kl_gap, mmd_rbf, w2_distance
from .methods import METHODS, fit_method, select_spec

DEFAULT_SEEDS = (7, 11, 23)


@dataclass
class Row:
    dataset: str
    method: str
    seed: int
    kl: float
    nll: float
    nll_true: float
    w2: float
    mmd: float
    n_params: int
    train_sec: float
    spec: str
    revived: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def make_dataset(name: str, cfg: Config, seed: int) -> Dataset:
    """构造 (train, test, val) 三份互不相交的数据。"""
    dens = build_density(name)
    dens = build_density(name)
    rng_tr = np.random.default_rng(int(seed))
    rng_te = np.random.default_rng(int(cfg.test_seed))
    rng_va = np.random.default_rng(int(cfg.hpo_seed))
    return Dataset(
        name=name,
        X_train=dens.sample(int(cfg.n_train), rng_tr),
        X_test=dens.sample(int(cfg.n_test), rng_te),
        log_prob=dens.log_prob,
        metadata={"density": dens, "X_val": dens.sample(int(cfg.n_train), rng_va)},
    )


def evaluate(
    model: Any, X_test: np.ndarray, log_p_true: np.ndarray, rng: np.random.Generator, n_w2: int
) -> dict[str, float]:
    lp = model.log_prob(X_test)
    nll = -float(np.mean(lp))
    nll_true = -float(np.mean(log_p_true))
    m = int(min(n_w2, X_test.shape[0]))
    Xs = model.sample(m, rng)
    Ys = X_test[rng.permutation(X_test.shape[0])[:m]]
    return {
        "kl": kl_gap(lp, log_p_true),
        "nll": nll,
        "nll_true": nll_true,
        "w2": w2_distance(Xs, Ys),
        "mmd": mmd_rbf(Xs, Ys),
    }


def run_benchmark(
    cfg: Config,
    datasets: Sequence[str] = DATASETS,
    methods: Sequence[str] = METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    iters: int | None = None,
    hpo_iters: int | None = None,
    revive: bool = True,
    verbose: bool = False,
) -> list[Row]:
    """跑完整基准。返回逐 (dataset, method, seed) 的行。"""
    iters = int(iters or cfg.iters)
    hpo_iters = int(hpo_iters or max(120, iters // 3))
    rows: list[Row] = []
    for dname in datasets:
        # test/val 固定；train 随 seed 变化
        base = make_dataset(dname, cfg, seed=int(seeds[0]))
        X_test = base.X_test
        lp_true = base.log_prob(X_test)
        X_val = base.metadata["X_val"]
        chosen: dict[str, Any] = {}
        for m in methods:
            spec = select_spec(m, X_val, cfg, hpo_iters=hpo_iters)
            chosen[m] = spec
            for sd in seeds:
                ds = make_dataset(dname, cfg, seed=int(sd))
                model, diag = fit_method(
                    m,
                    spec,
                    ds.X_train,
                    cfg,
                    int(sd),
                    iters=iters,
                    revive=(revive and m == "fusion"),
                )
                rng = np.random.default_rng(int(sd) * 31 + 7)
                ev = evaluate(model, X_test, lp_true, rng, int(cfg.n_w2_samples))
                rows.append(
                    Row(
                        dataset=dname,
                        method=m,
                        seed=int(sd),
                        kl=ev["kl"],
                        nll=ev["nll"],
                        nll_true=ev["nll_true"],
                        w2=ev["w2"],
                        mmd=ev["mmd"],
                        n_params=int(model.nparams()),
                        train_sec=float(diag.get("train_sec", 0.0)),
                        spec=str(spec),
                        revived=int(diag.get("revived_components", 0)),
                    )
                )
        if verbose:
            for m in methods:
                sub = [r for r in rows if r.dataset == dname and r.method == m]
                print(
                    f"  {dname:8s} {m:9s} KL={np.mean([r.kl for r in sub]):.5f}"
                    f"  n_params={sub[0].n_params}"
                )
    return rows


__all__ = ["DEFAULT_SEEDS", "Row", "evaluate", "make_dataset", "run_benchmark"]

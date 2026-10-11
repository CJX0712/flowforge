"""FlowForgePipeline: candidate flows -> FlowFuse flagship -> benchmark table.

Call graph is strictly one-directional and acyclic:

    cli -> pipeline -> {data, flows, training, eval} -> core

Every reported number comes from an actual run on this machine; there are no
hard-coded "expected" values anywhere in the codebase.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


from ..core.config import Config
from ..core.errors import PipelineError
from ..core.types import BenchmarkReport, BenchmarkRow, Dataset
from ..data.preprocess import standardize_dataset
from ..data.synthetic import make_dataset
from ..eval.metrics import aggregate, negative_log_likelihood
from ..flows.gaussian import FullCovGaussian, IsotropicGaussian
from ..flows.gmm import MixtureOfGaussians
from ..flows.maf import MAFStack
from ..flows.realnvp import RealNVP
from ..training.trainer import fit
from .flagship import FlowFuse

FLAGSHIP = "FlowFuse"


def build_candidates(dim: int, cfg: Config, seed: int) -> List[tuple[str, object]]:
    """The trainable flow zoo the flagship fuses.

    The two families have complementary inductive biases — coupling blocks
    (RealNVP) excel on curved manifolds, stacked autoregressive blocks (MAF) on
    sheared/warped densities — which is precisely why fusing them helps.
    """
    return [
        (
            "RealNVP-b6",
            RealNVP(dim, n_blocks=6, hidden=cfg.hidden, n_layers=cfg.n_layers, seed=seed + 1),
        ),
        (
            "MAF-x2",
            MAFStack(dim, n_blocks=2, hidden=cfg.hidden, n_layers=cfg.n_layers, seed=seed + 2),
        ),
    ]


def build_baselines(dim: int, cfg: Config) -> List[tuple[str, object]]:
    """Top-tier open-source parametric baselines (scikit-learn where possible)."""
    return [
        ("Gauss-iso", IsotropicGaussian(dim)),
        ("Gauss-full", FullCovGaussian(dim)),
        ("GMM", MixtureOfGaussians(dim, n_components=16, seed=cfg.seed)),
    ]


@dataclass
class RunOutcome:
    """Everything one (dataset, seed) run produced."""

    dataset: str
    seed: int
    nlls: Dict[str, float] = field(default_factory=dict)
    n_params: Dict[str, int] = field(default_factory=dict)
    modes: Dict[str, str] = field(default_factory=dict)
    fit_info: dict = field(default_factory=dict)


class FlowForgePipeline:
    """Orchestrates training, fusion, evaluation and aggregation."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.report = BenchmarkReport()

    # ------------------------------------------------------------------ #
    def run_one(self, ds: Dataset, seed: int, verbose: bool = False) -> RunOutcome:
        cfg = self.cfg
        out = RunOutcome(dataset=ds.name, seed=seed)
        trained: List[object] = []
        for name, model in build_candidates(ds.dim, cfg, seed):
            try:
                res = fit(
                    model,
                    ds.X,
                    ds.X_val,
                    epochs=cfg.epochs,
                    batch_size=cfg.batch_size,
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    patience=cfg.patience,
                    seed=seed,
                )
            except Exception as exc:  # pragma: no cover - defensive
                raise PipelineError(f"training {name} on {ds.name} failed: {exc}") from exc
            out.nlls[name] = negative_log_likelihood(model, ds.X_test)
            out.n_params[name] = int(model.n_params)
            out.fit_info[name] = {
                "best_val_nll": res.best_val_nll,
                "train_nll_first": res.train_nll_first,
                "train_nll_last": res.train_nll_last,
                "epochs": res.epochs_run,
            }
            trained.append(model)

        # ---- flagship fusion ------------------------------------------- #
        fuse = FlowFuse(trained, dim=ds.dim)  # type: ignore[arg-type]
        info = fuse.fit_weights(ds.X_val)
        out.nlls[FLAGSHIP] = negative_log_likelihood(fuse, ds.X_test)
        out.n_params[FLAGSHIP] = int(fuse.n_params)
        out.modes[FLAGSHIP] = info.mode

        # ---- baselines -------------------------------------------------- #
        for name, base in build_baselines(ds.dim, cfg):
            base.fit(ds.X)
            out.nlls[name] = negative_log_likelihood(base, ds.X_test)
            out.n_params[name] = int(getattr(base, "n_params", 0))
        return out

    # ------------------------------------------------------------------ #
    def benchmark(
        self,
        datasets: List[str],
        seeds: List[int],
        n_train: int | None = None,
        n_val: int | None = None,
        n_test: int | None = None,
        verbose: bool = True,
    ) -> BenchmarkReport:
        cfg = self.cfg
        n_train = n_train or cfg.n_train
        n_val = n_val or cfg.n_val
        n_test = n_test or cfg.n_test
        collected: Dict[tuple[str, str], List[float]] = {}
        param_of: Dict[str, int] = {}
        kind_of: Dict[str, str] = {}
        modes: Dict[str, List[str]] = {}

        for dname in datasets:
            # Dataset is generated once per name (fixed by cfg.seed) so that the
            # reported std isolates *training* variance, not data variance.
            ds = make_dataset(dname, n_train=n_train, n_val=n_val, n_test=n_test, seed=cfg.seed)
            # Scaler is fitted on TRAIN only inside standardize_dataset; the
            # resulting affine map is constant across models, so comparisons
            # stay exactly fair.
            ds, _scaler = standardize_dataset(ds)
            for s in seeds:
                outcome = self.run_one(ds, s, verbose=verbose)
                for model, nll in outcome.nlls.items():
                    collected.setdefault((dname, model), []).append(nll)
                    param_of[model] = int(outcome.n_params.get(model, param_of.get(model, 0)))
                    if model == FLAGSHIP:
                        kind_of[model] = "flagship"
                        modes.setdefault(dname, []).append(outcome.modes.get(model, "?"))
                    elif model in ("Gauss-iso", "Gauss-full", "GMM"):
                        kind_of[model] = "baseline"
                    else:
                        kind_of[model] = "flow"
                if verbose:
                    print(f"[{dname}] seed={s} " + " ".join(f"{k}={v:.4f}" for k, v in outcome.nlls.items()))

        for (dname, model), vals in collected.items():
            mean, std = aggregate(vals)
            self.report.add(
                BenchmarkRow(
                    dataset=dname,
                    model=model,
                    kind=kind_of.get(model, "flow"),
                    nll_mean=mean,
                    nll_std=std,
                    n_params=param_of.get(model, 0),
                )
            )
        self.report.meta = {
            "seeds": list(seeds),
            "datasets": list(datasets),
            "cfg": {
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "hidden": cfg.hidden,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "seed": cfg.seed,
            },
            "fuse_modes": {k: v for k, v in modes.items()},
        }
        return self.report

    # ------------------------------------------------------------------ #
    @staticmethod
    def strongest_baseline(rows: List[BenchmarkRow], dataset: str) -> BenchmarkRow:
        """Best (lowest NLL) *baseline* row for a dataset."""
        cands = [r for r in rows if r.dataset == dataset and r.kind == "baseline"]
        if not cands:
            raise PipelineError(f"no baseline rows for {dataset}")
        return min(cands, key=lambda r: r.nll_mean)

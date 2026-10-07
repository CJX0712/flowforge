"""配置：dataclass + FLOWFORGE_* 环境变量覆盖 + schema 校验。

踩坑库 §A：配置非法值必须抛错，不能静默兜底（静默兜底会把非法输入伪装成"降级成功"）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any

from .errors import ConfigError

ENV_PREFIX = "FLOWFORGE_"

_INT_FIELDS = {
    "seed",
    "n_train",
    "n_test",
    "iters",
    "batch",
    "hidden",
    "n_blocks",
    "hpo_seed",
    "n_bins",
    "n_components",
}
_FLOAT_FIELDS = {"lr", "entropy_reg", "grad_clip", "w2_reg"}


@dataclass
class Config:
    # 确定性
    seed: int = 7
    hpo_seed: int = 9042
    test_seed: int = 10007
    # 数据
    n_train: int = 2000
    n_test: int = 2000
    # 训练：默认预算经实测标定，使 demo 端到端 <= 60s（CPU）
    iters: int = 250
    lr: float = 5e-3
    batch: int = 256
    grad_clip: float = 5.0
    # 结构
    hidden: int = 32
    n_blocks: int = 8
    n_bins: int = 8
    n_components: int = 4
    entropy_reg: float = 0.0
    # 评测
    w2_reg: float = 0.05
    n_w2_samples: int = 200

    @classmethod
    def from_env(cls, **overrides: Any) -> Config:
        data: dict[str, Any] = {}
        for f in fields(cls):
            env_key = ENV_PREFIX + f.name.upper()
            if env_key in os.environ:
                raw = os.environ[env_key]
                try:
                    data[f.name] = int(raw) if f.name in _INT_FIELDS else float(raw)
                except ValueError as exc:
                    raise ConfigError(f"环境变量 {env_key} 无法解析为数字", value=raw) from exc
        data.update(overrides)
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError("seed 必须 >= 0", seed=self.seed)
        if self.n_train <= 0 or self.n_test <= 0:
            raise ConfigError("n_train/n_test 必须 > 0", n_train=self.n_train, n_test=self.n_test)
        if self.iters <= 0:
            raise ConfigError("iters 必须 > 0", iters=self.iters)
        if self.lr <= 0:
            raise ConfigError("lr 必须 > 0", lr=self.lr)
        if self.batch <= 0 or self.batch > max(self.n_train, 1):
            raise ConfigError(
                "batch 必须在 (0, n_train] 内", batch=self.batch, n_train=self.n_train
            )
        if self.hidden <= 0 or self.n_blocks <= 0:
            raise ConfigError(
                "hidden/n_blocks 必须 > 0", hidden=self.hidden, n_blocks=self.n_blocks
            )
        if self.n_bins < 2:
            raise ConfigError("n_bins 必须 >= 2", n_bins=self.n_bins)
        if self.n_components < 1:
            raise ConfigError("n_components 必须 >= 1", n_components=self.n_components)
        if self.w2_reg <= 0:
            raise ConfigError("w2_reg 必须 > 0", w2_reg=self.w2_reg)


__all__ = ["ENV_PREFIX", "Config"]

"""FlowForge 错误码（E1xx 配置 / E2xx 数据 / E3xx 流 / E4xx 训练 / E5xx 评测）。

约定：所有可预期的失败都必须抛 FlowForgeError 子类并带稳定错误码，
禁止用裸 Exception 或静默兜底（踩坑库 §A：守卫没被测试覆盖 = 带 bug 提交）。
"""

from __future__ import annotations


class FlowForgeError(Exception):
    """所有 FlowForge 异常的基类。"""

    code = "E000"

    def __init__(self, message: str, **ctx: object) -> None:
        self.ctx = ctx
        detail = f"[{self.code}] {message}"
        if ctx:
            detail += " | " + ", ".join(f"{k}={v}" for k, v in ctx.items())
        super().__init__(detail)


# ---- E1xx 配置 ------------------------------------------------------------
class ConfigError(FlowForgeError):
    code = "E100"


class UnknownMethodError(ConfigError):
    code = "E101"


class UnknownDatasetError(ConfigError):
    code = "E102"


# ---- E2xx 数据 ------------------------------------------------------------
class DataError(FlowForgeError):
    code = "E200"


class ShapeMismatchError(DataError):
    code = "E201"


class DensityNotNormalizedError(DataError):
    code = "E202"


# ---- E3xx 流 / 双射 -------------------------------------------------------
class FlowError(FlowForgeError):
    code = "E300"


class NotInvertibleError(FlowError):
    code = "E301"


class LogDetMismatchError(FlowError):
    code = "E302"


class GradientCheckError(FlowError):
    code = "E303"


# ---- E4xx 训练 ------------------------------------------------------------
class TrainingError(FlowError):
    code = "E400"


class DivergenceError(TrainingError):
    code = "E401"


# ---- E5xx 评测 ------------------------------------------------------------
class EvalError(FlowForgeError):
    code = "E500"


class NegativeKLError(EvalError):
    code = "E501"


__all__ = [
    "ConfigError",
    "DataError",
    "DensityNotNormalizedError",
    "DivergenceError",
    "EvalError",
    "FlowError",
    "FlowForgeError",
    "GradientCheckError",
    "LogDetMismatchError",
    "NegativeKLError",
    "NotInvertibleError",
    "ShapeMismatchError",
    "TrainingError",
    "UnknownDatasetError",
    "UnknownMethodError",
]

"""数据层：可解析求对数密度的合成分布。"""

from .densities import DATASETS, Banana, Funnel, GaussianMixture, Swirl, build_density

__all__ = ["DATASETS", "Banana", "Funnel", "GaussianMixture", "Swirl", "build_density"]

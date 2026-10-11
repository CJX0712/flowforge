"""FlowForge flows package: invertible transforms and density models.

Tier-1 (always available, zero-download): hand-written NumPy RealNVP, MAF,
Gaussian / GMM baselines.  Optional Tier-0 SOTA (``nflows``/``torch``) is
probed at runtime and gracefully skipped.

Author: 晨星
"""

from __future__ import annotations

from .base import AbstractFlow, base_logprob, sample_base
from .composite import CompositeFlow
from .coupling import AffineCoupling
from .gaussian import FullCovGaussian, IsotropicGaussian
from .gmm import MixtureOfGaussians
from .maf import MaskedAutoregressiveFlow, MaskedMLP
from .mlp import MLP
from .permute import RandomPermutation
from .realnvp import RealNVP, half_masks, param_pointers

__all__ = [
    "AbstractFlow",
    "AffineCoupling",
    "CompositeFlow",
    "FullCovGaussian",
    "IsotropicGaussian",
    "MLP",
    "MaskedAutoregressiveFlow",
    "MaskedMLP",
    "MixtureOfGaussians",
    "RandomPermutation",
    "RealNVP",
    "base_logprob",
    "half_masks",
    "param_pointers",
    "sample_base",
]

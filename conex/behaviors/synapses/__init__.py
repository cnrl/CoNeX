"""Synapse behaviors for CoNeX."""

from . import numenta
from .specs import *
from .learning import *
from .dendrites import *
from .numenta import *

__all__ = [
    # Specs
    "SynapseInit",
    "DelayInitializer",
    "WeightInitializer",
    "WeightNormalization",
    "CurrentNormalization",
    "WeightClip",
    "PreSpikeCatcher",
    "PostSpikeCatcher",
    "PreTrace",
    "PostTrace",
    # Weight bounds
    "soft_bound",
    "hard_bound",
    "no_bound",
    "BOUNDS",
    # Learning
    "BaseLearning",
    "SimpleSTDP",
    "SparseSTDP",
    "One2OneSTDP",
    "SimpleiSTDP",
    "One2OneiSTDP",
    "SparseiSTDP",
    "Conv2dSTDP",
    "Local2dSTDP",
    "SimpleRSTDP",
    "One2OneRSTDP",
    "SparseRSTDP",
    "Conv2dRSTDP",
    "Local2dRSTDP",
    # Dendrites
    "BaseDendriticInput",
    "SparseDendriticInput",
    "One2OneDendriticInput",
    "SimpleDendriticInput",
    "AveragePool2D",
    "LateralDendriticInput",
    "Conv2dDendriticInput",
    "Local2dDendriticInput",
    # Numenta
    *numenta.__all__,
]

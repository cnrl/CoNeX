"""Neuron behaviors from Numenta's Thousand Brains Theory and HTM."""

from .grid_cells import GridCellModule, DisplacementCellModule, LocationEncoder
from .active_dendrites import (
    DendriticSegment,
    DendriticSegmentConfig,
    ActiveDendriteComputation,
    ContextualPrediction,
)
from .sequence_memory import TemporalMemory, TemporalMemoryConfig, SegmentStorage
from .spatial_pooler import SpatialPooler, SpatialPoolerConfig
from .sdr import (
    SDR,
    SDRConfig,
    SDREncoder,
    SDRClassifier,
    UnionSDR,
)
from .predictive_coding import (
    PredictionType,
    PredictiveCodingConfig,
    PredictionUnit,
    ErrorUnit,
    PrecisionWeighting,
    TopDownPrediction,
    PredictiveCodingLearning,
    FreeEnergyMinimization,
    HierarchicalPredictiveCoding,
    create_predictive_layer,
)

__all__ = [
    # Grid Cells
    "GridCellModule",
    "DisplacementCellModule",
    "LocationEncoder",
    # Active Dendrites
    "DendriticSegment",
    "DendriticSegmentConfig",
    "ActiveDendriteComputation",
    "ContextualPrediction",
    # Sequence Memory
    "TemporalMemory",
    "TemporalMemoryConfig",
    "SegmentStorage",
    # Spatial Pooler
    "SpatialPooler",
    "SpatialPoolerConfig",
    # SDR
    "SDR",
    "SDRConfig",
    "SDREncoder",
    "SDRClassifier",
    "UnionSDR",
    # Predictive Coding
    "PredictionType",
    "PredictiveCodingConfig",
    "PredictionUnit",
    "ErrorUnit",
    "PrecisionWeighting",
    "TopDownPrediction",
    "PredictiveCodingLearning",
    "FreeEnergyMinimization",
    "HierarchicalPredictiveCoding",
    "create_predictive_layer",
]

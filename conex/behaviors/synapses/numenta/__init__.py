"""Synapse behaviors for predictive coding connections."""

from .predictive import (
    PredictiveSynapseInit,
    FeedbackPredictionSynapse,
    FeedforwardErrorSynapse,
    LateralPredictionSynapse,
    PredictiveSynapseLearning,
    PredictiveConnection,
)

__all__ = [
    "PredictiveSynapseInit",
    "FeedbackPredictionSynapse",
    "FeedforwardErrorSynapse",
    "LateralPredictionSynapse",
    "PredictiveSynapseLearning",
    "PredictiveConnection",
]

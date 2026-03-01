"""
Behavior Package
----------------

Behavioral models used in:

Project 1 — Market Reality & Dynamics
Project 2 — Risk & Fat-Tail
Project 3 — Decision Under Uncertainty

Public API:
    - OverconfidenceModel
    - LossAversionModel
    - ConfirmationBiasModel
    - HerdingModel
    - PanicCascadeModel
"""

# Bias models
from .bias import (
    BehavioralBias,
    OverconfidenceModel,
    LossAversionModel,
    ConfirmationBiasModel,
    apply_bias_pipeline,
)

# Herding dynamics
from .herding import HerdingModel

# Panic cascade
from .panic import PanicCascadeModel


__all__ = [
    # Base
    "BehavioralBias",

    # Bias
    "OverconfidenceModel",
    "LossAversionModel",
    "ConfirmationBiasModel",
    "apply_bias_pipeline",

    # Herding
    "HerdingModel",

    # Panic
    "PanicCascadeModel",
]
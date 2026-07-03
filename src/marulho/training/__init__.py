"""MARULHO Training -- model, trainer, and developmental protocols.

Module structure:
- model.py:                MarulhoModel (all SNN layers, representation contract)
- trainer.py:              MarulhoTrainer (STDP loop, sleep, drift tracking)
- developmental_runner.py: 5-stage developmental protocol
- warm_bootstrap.py:       GloVe/teacher embedding -> prototype seeding
- autonomy_*.py:           Autonomous acquisition runners
- checkpointing.py:       Save/load checkpoint utilities
"""

from .bootstrap import PredictiveBootstrap
from .language_model import (
    LanguageBatch,
    LanguageModelConfig,
    LanguageSplit,
    MarulhoLanguageModel,
    MarulhoSelectiveSpikingStateBlock,
    build_language_model_splits,
    evaluate_language_model,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from .model import MarulhoModel
from .trainer import MarulhoTrainer

__all__ = [
    "LanguageBatch",
    "LanguageModelConfig",
    "LanguageSplit",
    "MarulhoLanguageModel",
    "MarulhoModel",
    "MarulhoSelectiveSpikingStateBlock",
    "MarulhoTrainer",
    "PredictiveBootstrap",
    "build_language_model_splits",
    "evaluate_language_model",
    "load_language_model_checkpoint",
    "save_language_model_checkpoint",
]

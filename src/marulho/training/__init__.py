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
from .language_continual_learning import (
    LanguageContinualLearningConfig,
    run_language_continual_learning_window,
)
from .language_model import (
    LanguageBatch,
    LanguageModelConfig,
    LanguageSplit,
    MarulhoLanguageModel,
    MarulhoSelectiveSpikingStateBlock,
    RoutedLanguageExpertLayer,
    build_language_model_splits,
    evaluate_language_model,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from .model import MarulhoModel
from .trainer import MarulhoTrainer

__all__ = [
    "LanguageBatch",
    "LanguageContinualLearningConfig",
    "LanguageModelConfig",
    "LanguageSplit",
    "MarulhoLanguageModel",
    "MarulhoModel",
    "MarulhoSelectiveSpikingStateBlock",
    "MarulhoTrainer",
    "PredictiveBootstrap",
    "RoutedLanguageExpertLayer",
    "build_language_model_splits",
    "evaluate_language_model",
    "load_language_model_checkpoint",
    "run_language_continual_learning_window",
    "save_language_model_checkpoint",
]

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
from .model import MarulhoModel
from .trainer import MarulhoTrainer

__all__ = ["PredictiveBootstrap", "MarulhoModel", "MarulhoTrainer"]

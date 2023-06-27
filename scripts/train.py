"""
This is the script for the training process.
"""

from datasets import load_dataset

import torch
from torch import nn

import pytorch_lightning as pl

# import dataloader and dataset class
from torch.utils.data import DataLoader, Dataset

# import the model
from deep_triangularization.models import MLP_dense, MLP_triangular




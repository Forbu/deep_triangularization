"""
This is the script for the training process.
"""

import sys
import os

# Add the path to the deep_triangularization package to the Python path
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, package_path)

import torch

import lightning as L
from deep_triangularization.models import MLP_triangular
from utils import (
    load_tabular_dataset,
    get_dataloader,
    TabularClassifier,
    compute_next_version,
    get_train_test_dataloader,
    init_dataset,
)


dir_log = "logs/"
dim_embedding = 3

# load the dataset
(
    train_dataloader,
    test_dataloader,
    mapping_categorical,
    mapping_continuous,
) = init_dataset(
    path_data="data/phpkIxskf.arff",
    name_class="Class",
    batch_size=256,
)

# define the model
model = MLP_triangular(
    in_dim=len(mapping_categorical) * dim_embedding + len(mapping_continuous),
    out_dim=2,
    hidden_dim=512,
    num_layers=4,
)


# define the classifier
classifier = TabularClassifier(
    model, mapping_categorical, mapping_continuous, dim_embedding=dim_embedding
)


version = compute_next_version(dir_log)

# we use a tensorbaord logger
logger = L.pytorch.loggers.TensorBoardLogger(
    "logs/", name="tabular_classifier_marketing_triangle", version=version
)

# define the trainer
trainer = L.Trainer(
    max_epochs=40,
    log_every_n_steps=20,
    logger=logger,
    gradient_clip_val=1.0,
    precision=16,
)

# train the model
trainer.fit(classifier, train_dataloader, test_dataloader)

# save the model (state_dict)
# model add the version
model_name = "model_triangle_{}.pt".format(version)
torch.save(classifier.model.state_dict(), model_name)

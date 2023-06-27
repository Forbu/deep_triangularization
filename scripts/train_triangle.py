"""
This is the script for the training process.
"""

import sys
import os

# Add the path to the deep_triangularization package to the Python path
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, package_path)

import lightning as L
from deep_triangularization.models import MLP_triangular
from utils import (
    load_tabular_dataset,
    get_dataloader,
    TabularClassifier,
    compute_next_version,
)

name_class = "Class"
dir_log = "logs/"
dim_embedding = 3

# load the dataset
df = load_tabular_dataset()


# we compute the mapping for the categorical features
mapping_categorical = {
    idx: df[feature].nunique()
    for idx, feature in enumerate(df.columns)
    if df[feature].dtype == "O" and feature != name_class
}

# we compute the mapping for the continuous features
mapping_continuous = {
    idx: df[feature].nunique()
    for idx, feature in enumerate(df.columns)
    if df[feature].dtype != "O" and feature != name_class
}

# divide df into train and test set

# get the dataloader
dataloader = get_dataloader(
    df,
    categorical_features=[
        feature for feature in df.columns if df[feature].dtype == "O"
    ],
)

# define the model
model = MLP_triangular(
    in_dim=len(mapping_categorical) * dim_embedding + len(mapping_continuous),
    out_dim=2,
    hidden_dim=256,
    num_layers=3,
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
    max_epochs=10,
    log_every_n_steps=10,
    logger=logger,
    gradient_clip_val=1.0,
)

# train the model
trainer.fit(classifier, dataloader)

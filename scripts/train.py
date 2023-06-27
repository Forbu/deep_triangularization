"""
This is the script for the training process.
"""

import sys
import os

# Add the path to the deep_triangularization package to the Python path
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, package_path)


import pandas as pd
from scipy.io import arff

# import labelencoder
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn

import pytorch_lightning as pl

# import dataloader and dataset class
from torch.utils.data import DataLoader, Dataset

# import the model
from deep_triangularization.models import MLP_dense, MLP_triangular


def load_tabular_dataset(path="data/dataset_31_credit-g.arff"):
    """
    Datase coming from https://www.openml.org/search?type=data&sort=nr_of_downloads&status=active&id=31
    """
    # load the dataset
    data, meta = arff.loadarff(path)

    # convert to pandas dataframe
    df = pd.DataFrame(data)

    return df


# define Dataset class
class TabularDataset(Dataset):
    def __init__(self, df, categorical_features):
        self.df = df

        # get the features
        self.features = df.columns[:-1]

        # get the target
        self.target = df.columns[-1]

        # we use a label encoder to encode the categorical features
        self.label_encoder = LabelEncoder()

        # we encode the categorical features
        for feature in categorical_features:
            self.df[feature] = self.label_encoder.fit_transform(self.df[feature])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # get the features
        x = torch.tensor(row[self.features].values).float()

        # get the target
        y = torch.tensor(row[self.target]).long()

        return x, y


# define the dataloader
def get_dataloader(df, batch_size=32, num_workers=0, categorical_features=None):
    dataset = TabularDataset(df, categorical_features)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return dataloader


# now we pytorch lightning we can define the training loop
class TabularClassifier(pl.LightningModule):
    def __init__(self, model, mapping_categorical, mapping_continuous, dim_embedding=3):
        super(TabularClassifier, self).__init__()
        self.model = model

        self.mapping_categorical = mapping_categorical
        self.mapping_continuous = mapping_continuous

        self.index_categorical = list(mapping_categorical.keys())

        self.index_continuous = list(mapping_continuous.keys())

        # we create as many embedding layers as there are categorical features
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(mapping_categorical[feature], dim_embedding)
                for feature in mapping_categorical
            ]
        )

        # batch normalization
        self.batch_norm = nn.BatchNorm1d(
            len(self.index_categorical) * dim_embedding + len(self.index_continuous)
        )

        # define the loss function
        self.loss = nn.CrossEntropyLoss()

    def forward(self, continuous, categorical):
        # we apply the embedding layers to the categorical features

        x = torch.cat(
            [
                embedding(categorical[:, i])
                for i, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )

        # we apply the batch normalization
        x = self.batch_norm(torch.cat([x, continuous], dim=1))

        return self.model(x)

    def compute_loss(self, batch):
        x, y = batch

        # get the categorical and continuous features
        categorical = x[:, self.index_categorical].long()

        continuous = x[:, self.index_continuous].float()

        # get the predictions
        y_hat = self.forward(continuous, categorical)

        # compute the loss
        loss = self.loss(y_hat, y)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# load the dataset
df = load_tabular_dataset()

print(df.head())

print(df.dtypes)

print(df.shape)

# we compute the mapping for the categorical features
mapping_categorical = {
    idx: df[feature].nunique()
    for idx, feature in enumerate(df.columns)
    if df[feature].dtype == "O" and feature != "class"
}

# we compute the mapping for the continuous features
mapping_continuous = {
    idx: df[feature].nunique()
    for idx, feature in enumerate(df.columns)
    if df[feature].dtype != "O" and feature != "class"
}

# get the dataloader
dataloader = get_dataloader(
    df,
    categorical_features=[
        feature for feature in df.columns if df[feature].dtype == "O"
    ],
)

# test dataloader
for batch in dataloader:
    x, y = batch

    print(x.shape, y.shape)

    break

####### First model: MLP with dense layers #######

dim_embedding = 3

# define the model
model = MLP_dense(
    in_dim=len(mapping_categorical) * dim_embedding + len(mapping_continuous),
    out_dim=2,
    hidden_dim=128,
    num_layers=5,
)


# define the classifier
classifier = TabularClassifier(
    model, mapping_categorical, mapping_continuous, dim_embedding=dim_embedding
)

# we use a tensorbaord logger
logger = pl.loggers.TensorBoardLogger("logs/")

# define the trainer
trainer = pl.Trainer(
    max_epochs=10,
    logger=logger,
    gradient_clip_val=1.0,
)

# train the model
trainer.fit(classifier, dataloader)

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

import lightning as L

# import dataloader and dataset class
from torch.utils.data import DataLoader, Dataset

# import the model
from deep_triangularization.models import MLP_dense, MLP_triangular


def load_tabular_dataset(path="data/phpkIxskf.arff"):
    """
    Datase coming from https://www.openml.org/search?type=data&sort=nr_of_downloads&status=active&id=31
    or other
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
def get_dataloader(df, batch_size=256, num_workers=0, categorical_features=None):
    dataset = TabularDataset(df, categorical_features)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return dataloader


# now we pytorch lightning we can define the training loop
class TabularClassifier(L.LightningModule):
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

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        self.log("validation_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# load the dataset
df = load_tabular_dataset()

print(df.head())

print(df.dtypes)

print(df.shape)

name_class = "Class"

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

# test dataloader
for batch in dataloader:
    x, y = batch

    print(x.shape, y.shape)

    break

####### First model: MLP with dense layers #######

dim_embedding = 3

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

dir_log = "logs/"

from glob import glob

# list all the directories and subdirectories in dir_log
# we look at the version_XX/ dir name. We look at the highest XX and set the version to XX + 1
list_dir = glob(dir_log + "**", recursive=True)

# filter the list to keep only the directories
list_dir = [dir for dir in list_dir if os.path.isdir(dir)]

# now we want the basename of each dir to be able to filter the version_XX/ dir name
list_dir = [os.path.basename(dir) for dir in list_dir]

# now we look at the version_XX/ dir name
list_version = [
    int(dir.split("/")[-1].split("_")[1]) for dir in list_dir if "version_" in dir
]

# we set the version to the highest version + 1
version = max(list_version) + 1

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

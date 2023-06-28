"""
Script to anaylize the models wieghts
"""

import torch

from utils import load_tabular_dataset, get_train_test_dataloader, get_dataloader, TabularClassifier

path_model_1 = "model_triangle_29.pt"
path_model_2 = "model_triangle_30.pt"

# load the model
model_1 = torch.load(path_model_1)
model_2 = torch.load(path_model_2)

print(model_1["layers.1.weight"] * model_1["layers.1.mask"] - model_2["layers.1.weight"] * model_2["layers.1.mask"])
print(torch.mean(torch.abs(model_1["layers.1.weight"] * model_1["layers.1.mask"] - model_2["layers.1.weight"] * model_2["layers.1.mask"])))

path_model_1 = "model_dense_33.pt"
path_model_2 = "model_dense_34.pt"

# load the model
model_1 = torch.load(path_model_1)
model_2 = torch.load(path_model_2)

print(model_1["layers.1.weight"] - model_2["layers.1.weight"])
print(torch.mean(torch.abs(model_1["layers.1.weight"] - model_2["layers.1.weight"])))


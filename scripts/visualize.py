"""
Script to visualize the loss function in 2D
We use the method in the paper : Visualizing the Loss Landscape of Neural Nets (https://arxiv.org/pdf/1712.09913.pdf)
"""
import torch

from utils import (
    load_tabular_dataset,
    get_train_test_dataloader,
    get_dataloader,
    TabularClassifier,
    init_dataset,
)

# so we have to take one saved model and visualize the loss function in 2D
path_model_triangle = "model_triangle_12.pt"

# we load the model
model = torch.load(path_model_triangle)

# we select the second layer
weights_1 = model["layers.1.weight"]
weights_2 = model["layers.2.weight"]

shape_1 = weights_1.shape
shape_2 = weights_2.shape
n = 5
m = 5

# we generate n random gaussien noise with the same shape as the weights
noise_1 = torch.randn(n, shape_1[0], shape_1[1])
noise_2 = torch.randn(n, shape_2[0], shape_2[1])

# now we apply a normalization to the noise
noise_1 = noise_1 / torch.norm(noise_1, dim=0, keepdim=True)
noise_2 = noise_2 / torch.norm(noise_2, dim=0, keepdim=True)

# now we normalize with the weights
noise_1 = noise_1 * torch.norm(weights_1)
noise_2 = noise_2 * torch.norm(weights_2)

# now we create the loss function
loss_function = torch.nn.CrossEntropyLoss()

# we load the dataset
(
    train_dataloader,
    test_dataloader,
    mapping_categorical,
    mapping_continuous,
) = init_dataset(
    path_data="data/phpkIxskf.arff",
    name_class="Class",
    batch_size=4096,
)

index_categorical = list(mapping_categorical.keys())
index_continuous = list(mapping_continuous.keys())

# we extract the first batch
for batch in train_dataloader:
    x, y = batch
    categorical = x[:, index_categorical].long()
    continuous = x[:, index_continuous].float()

    break

# we compute the loss for each noise
dx_1 = 0.05
dx_2 = 0.05

# now we compute a meshgrid around the weights
x_1, x_2 = torch.meshgrid(
    torch.arange(
        -2.0,
        2.0,
        dx_1,
    ),
    torch.arange(
        -2.0,
        2.0,
        dx_2,
    ),
)

# we compute the loss for each point in the meshgrid
losses = torch.zeros(x_1.shape)


def compute_loss(model, x_1, x_2, categorical, continuous):
    model["layers.1.weight"] = weights_1 + x_1 * noise_1[0, :, :]
    model["layers.2.weight"] = weights_2 + x_2 * noise_2[0, :, :]

    loss = model.compute_loss(
        (torch.cat([continuous] * n), torch.cat([categorical] * n))
    )

    return loss.item()


# we loop over the meshgrid and compute the loss for each point
for i in range(x_1.shape[0]):
    for j in range(x_1.shape[1]):
        losses[i, j] = compute_loss(
            model, x_1[i, j], x_2[i, j], categorical, continuous
        )

# we plot the loss function
import matplotlib.pyplot as plt

plt.contourf(x_1, x_2, losses, 100)
plt.colorbar()

# save the figure
plt.savefig("loss_function_triangle.png")

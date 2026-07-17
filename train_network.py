from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from neural_model import CerebellumNet


HERE = Path(__file__).parent

dataset = pd.read_csv(
    HERE / "dataset.csv"
)

# ----------------------------------------------------
# Inputs
# ----------------------------------------------------

X = dataset[[
    "q1",
    "q2",
    "dq1",
    "dq2",
    "desired_q1",
    "desired_q2"
]].values.astype(np.float32)

# ----------------------------------------------------
# Outputs
# ----------------------------------------------------

Y = dataset[[
    "tau1",
    "tau2"
]].values.astype(np.float32)

# ----------------------------------------------------
# Normalization
# ----------------------------------------------------

X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8

Y_mean = Y.mean(axis=0)
Y_std = Y.std(axis=0) + 1e-8

# Ensure models directory exists BEFORE saving normalization files
models_dir = HERE / "models"
models_dir.mkdir(parents=True, exist_ok=True)

np.save(models_dir / "input_mean.npy", X_mean)
np.save(models_dir / "input_std.npy", X_std)

np.save(models_dir / "output_mean.npy", Y_mean)
np.save(models_dir / "output_std.npy", Y_std)

# ----------------------------------------------------
# Torch tensors
# ----------------------------------------------------

X = (X - X_mean) / X_std
Y = (Y - Y_mean) / Y_std

X = torch.tensor(X)
Y = torch.tensor(Y)

dataset = TensorDataset(X, Y)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

# ----------------------------------------------------
# Network
# ----------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = CerebellumNet().to(device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    network.parameters(),
    lr=1e-3
)

# ----------------------------------------------------
# Training
# ----------------------------------------------------

EPOCHS = 80

print()
for epoch in range(EPOCHS):
    running = 0.0
    network.train()
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = network(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        running += loss.item()

    avg_loss = running / len(loader)
    print(f"Epoch {epoch+1:03d} | Loss = {avg_loss:.6f}")

# ----------------------------------------------------
# Save model
# ----------------------------------------------------

torch.save(
    network.state_dict(),
    models_dir / "cerebellum.pth"
)

print()
print("Training complete.")
print(f"Model saved to: {models_dir / 'cerebellum.pth'}")
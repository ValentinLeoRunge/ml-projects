import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# ---------------------- model -----------------------
#prepare data
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)
n_samples, n_features = x.shape

input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# ---------------- loss and optimizer ----------------
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

# ------------------ training loop -------------------
num_epochs = 100
for epoch in range(num_epochs):
    y_predicted = model(x)
    loss = criterion(y_predicted, y)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

predicted = model(x).detach().numpy()

plt.figure(figsize=(10, 6))
plt.plot(x_numpy, y_numpy, "ro", label="Training Data (x, y)")
plt.plot(x_numpy, predicted, "b", label=f"Model: y_hat = {model.weight.item():.2f}x + {model.bias.item():.2f}")
plt.xlabel("x (Input Feature)")
plt.ylabel("y (Output)")
plt.title(f"Linear Regression after {num_epochs} Epochs | MSE Loss = {loss.item():.2f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
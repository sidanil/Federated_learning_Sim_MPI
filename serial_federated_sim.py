import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
from sklearn.datasets import make_moons

# ----- Parse Args -----
parser = argparse.ArgumentParser()
parser.add_argument('--total-devices', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--rounds', type=int, default=10)
parser.add_argument('--samples', type=int, default=100000)
args = parser.parse_args()

# ----- Model & Utils -----
class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def get_model_weights(model):
    return [p.data.clone() for p in model.parameters()]

def set_model_weights(model, weights):
    for param, w in zip(model.parameters(), weights):
        param.data.copy_(w)

def average_weights(weight_list):
    return [torch.stack(weights).mean(0) for weights in zip(*weight_list)]

# ----- Dataset -----
X, y = make_moons(n_samples=args.samples, noise=0.2)
X_splits = np.array_split(X, args.total_devices)
y_splits = np.array_split(y, args.total_devices)

# ----- Training -----
start = time.time()
global_model = SimpleANN()

for _ in range(args.rounds):
    all_weights = []

    for i in range(args.total_devices):
        model = SimpleANN()
        set_model_weights(model, get_model_weights(global_model))
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        x_tensor = torch.tensor(X_splits[i], dtype=torch.float32)
        y_tensor = torch.tensor(y_splits[i], dtype=torch.long)

        for _ in range(args.epochs):
            optimizer.zero_grad()
            loss = criterion(model(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()

        all_weights.append(get_model_weights(model))

    set_model_weights(global_model, average_weights(all_weights))

end = time.time()
print(f"[SERIAL] {args.total_devices} devices | Time: {end - start:.2f} sec")

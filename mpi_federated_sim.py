from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from sklearn.datasets import make_moons

# ----- MPI Init -----
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----- Parse Args -----
parser = argparse.ArgumentParser()
parser.add_argument('--devices-per-rank', type=int, default=1)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--rounds', type=int, default=10)
parser.add_argument('--samples', type=int, default=100000)
args = parser.parse_args()

# Derived
total_devices = size * args.devices_per_rank

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

# ----- Dataset Distribution -----
if rank == 0:
    X, y = make_moons(n_samples=args.samples, noise=0.2)
    flat_X = np.array_split(X, total_devices)
    flat_y = np.array_split(y, total_devices)
    
    # Group chunks for each rank
    X_splits = [flat_X[i * args.devices_per_rank : (i + 1) * args.devices_per_rank] for i in range(size)]
    y_splits = [flat_y[i * args.devices_per_rank : (i + 1) * args.devices_per_rank] for i in range(size)]
else:
    X_splits = None
    y_splits = None

X_chunks = comm.scatter(X_splits, root=0)
y_chunks = comm.scatter(y_splits, root=0)

# ----- Federated Training -----
start = MPI.Wtime()
global_model = SimpleANN()

for _ in range(args.rounds):
    local_weights = []

    for i in range(args.devices_per_rank):
        X_local = torch.tensor(X_chunks[i], dtype=torch.float32)
        y_local = torch.tensor(y_chunks[i], dtype=torch.long)

        model = SimpleANN()
        set_model_weights(model, get_model_weights(global_model))
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        for _ in range(args.epochs):
            optimizer.zero_grad()
            loss = criterion(model(X_local), y_local)
            loss.backward()
            optimizer.step()

        local_weights.append(get_model_weights(model))

    gathered_weights = comm.gather(local_weights, root=0)

    if rank == 0:
        flat_weights = [w for group in gathered_weights for w in group]
        avg_weights = average_weights(flat_weights)
    else:
        avg_weights = None

    avg_weights = comm.bcast(avg_weights, root=0)
    set_model_weights(global_model, avg_weights)

end = MPI.Wtime()
if rank == 0:
    print(f"[MPI] {total_devices} devices over {size} ranks | Time: {end - start:.2f} sec")

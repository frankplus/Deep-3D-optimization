import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read positions
def generate_data(train_window = 12, train_set = 0.8):
    with open("positions_walk.txt", 'r') as f:
        positions = list()
        for line in f.readlines():
            pos = [float(x) for x in line.split()[:3]]
            positions.append(pos)

    positions = torch.FloatTensor(positions)

    dataset = []
    L = len(positions)
    for i in range(L-train_window):
        train_seq = positions[i:i+train_window]
        train_label = positions[i+train_window:i+train_window+1]
        dataset.append((train_seq ,train_label))

    train_samples = int(len(dataset) * train_set)
    training_set = dataset[:train_samples]
    eval_set = dataset[train_samples:]
    return training_set, eval_set


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12*3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train(data, model, loss_fn, optimizer, print_example=False):

    train_loss = 0
    for seq, label in data:
        # Compute prediction error
        input = seq.flatten()
        target = label.flatten()
        pred = model(input)
        loss = loss_fn(pred, target)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data)
    print(f"Train avg loss: {train_loss:>8f}")

    if print_example:
        print("example prediction: " + str(pred))
        print("example label: " + str(target))



def eval(data, model, loss_fn, print_example=False):

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for seq, label in data:
            input = seq.flatten()
            target = label.flatten()
            pred = model(input)
            loss = loss_fn(pred, target)
            test_loss += loss.item()
    test_loss /= len(data)
    print(f"Eval avg loss: {test_loss:>8f} \n")

    if print_example:
        print("example prediction: " + str(pred))
        print("example label: " + str(target))

def main():
    training_set, eval_set = generate_data()
    print(f"train samples: {len(training_set)}")
    print(f"eval samples: {len(eval_set)}")

    model = NeuralNetwork()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    print(model)

    # training
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    n_epochs = 100

    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_set, model, loss_fn, optimizer)
        eval(eval_set, model, loss_fn, t%20==0)

if __name__ == "__main__":
    main()  
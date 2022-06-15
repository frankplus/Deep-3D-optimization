from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

EPOCHS = 100

class SsimDataset(Dataset):
    def __init__(self):
        with open("../dynamic_ssim_multimodel/dataset.json", 'r') as f:
            data = json.load(f)
        self.dataset = list()
        for sample in data["samples"]:
            pos_ref = sample["pos_ref"]
            lod_id = data["lod_names"].index(sample["lod_name"])
            fps = sample["fps"] / 100.0
            self.dataset.append({"input": np.array(pos_ref + [lod_id]), "output": np.array([fps])})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, dropout = 0):
        super(NeuralNetwork, self).__init__()
        hidden_nodes = 256
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_nodes, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    train_loss = 0
    for batch, data in enumerate(dataloader):
        input = data["input"].float()
        label = data["output"].float()
        
        X, y = input.to(device), label.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= size
    print(f"Train avg loss: {train_loss:>8f}")
    return train_loss

def eval(dataloader, model, loss_fn, device, print_example=False):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            input = data["input"].float()
            label = data["output"].float()
            
            X, y = input.to(device), label.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= size
    print(f"Eval avg loss: {test_loss:>8f} \n")

    if print_example:
        print("example input: " + str(input))
        print("example prediction: " + str(pred))
        print("example label: " + str(label))
    return test_loss

def main():
    dataset = SsimDataset()

    print("example sample")
    print(dataset.__getitem__(0))

    train_samples = int(dataset.__len__() * 0.8)
    eval_samples = dataset.__len__() - train_samples
    train_set, eval_set = random_split(dataset, [train_samples, eval_samples])

    train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(eval_set, batch_size=4, shuffle=True)
    print(next(iter(train_dataloader)))

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = NeuralNetwork().to(device)
    print(model)

    # training
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.7)

    train_history = list()
    eval_history = list()
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        val_loss = eval(eval_dataloader, model, loss_fn, device, t%20==0)
        scheduler.step(val_loss)
        train_history.append(train_loss)
        eval_history.append(val_loss)
    print("Done!")

    # plot results
    plt.plot(train_history, label="train")
    plt.plot(eval_history, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
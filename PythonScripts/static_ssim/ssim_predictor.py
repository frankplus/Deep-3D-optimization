from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

EPOCHS = 200

with open("dataset.json", 'r') as f:
    JSON_DATA  = json.load(f)
LOD_NAMES = sorted(JSON_DATA["data"].keys())

class SsimDataset(Dataset):
    def __init__(self):
        self.dataset = []
        self.scaler = StandardScaler()
        normalized_positions = self.scaler.fit_transform(JSON_DATA["positions"])
        for i, position in enumerate(normalized_positions):
            position = np.array(position)
            sample_ssim = np.array([JSON_DATA["data"][lod][i]["ssim"] for lod in LOD_NAMES])
            sample = {"position": position, "ssim": sample_ssim}
            self.dataset.append(sample)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, dropout = 0.25):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(LOD_NAMES)),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    train_loss = 0
    for batch, data in enumerate(dataloader):
        input = data["position"].float()
        label = data["ssim"].float()
        
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
            input = data["position"].float()
            label = data["ssim"].float()
            
            X, y = input.to(device), label.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= size
    print(f"Eval avg loss: {test_loss:>8f} \n")

    if print_example:
        print("example prediction: " + str(pred))
        print("example label: " + str(label))
    return test_loss

def plot_surface(model, device, lod, scaler):
    def fun(x, y, z):
        normalized_input = scaler.transform([[x, y, z]])
        input = torch.from_numpy(normalized_input[0]).float().to(device)
        return model(input).detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5.0, 5.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x, 1.0, y)[lod] for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('SSIM')

    plt.show()

def main():
    dataset = SsimDataset()

    train_samples = int(dataset.__len__() * 0.8)
    eval_samples = int(dataset.__len__() * 0.2)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5)

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
    plot_surface(model, device, lod=len(LOD_NAMES)-1, scaler=dataset.scaler)

if __name__ == '__main__':
    main()
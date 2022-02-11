from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

EPOCHS = 100

class SubjectiveDataset(Dataset):
    def __init__(self):
        self.dataset = list()
        with open('experiment_results.txt') as json_file:
            data = json.load(json_file)
        
        for sample in data:
            ssim = sample["average_ssim"]
            fvc = sample["average_frame_vertex_count"]
            rating = sample["rating"] / 5.0
            self.dataset.append({"input": torch.FloatTensor([ssim, fvc]), 
                                "output": torch.FloatTensor([rating])})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, dropout = 0.1):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.ReLU()
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

def eval(dataloader, model, loss_fn, device, print_example=False, print_plot=False):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    labels = []
    predictions = []
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            input = data["input"].float()
            label = data["output"].float()
            
            X, y = input.to(device), label.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            labels += label.tolist()
            predictions += pred.tolist()
    test_loss /= size
    print(f"Eval avg loss: {test_loss:>8f} \n")

    if print_example:
        print("example input: " + str(input))
        print("example prediction: " + str(pred))
        print("example label: " + str(label))

    if print_plot:
        plt.scatter(predictions, labels, s=20)
        plt.xlabel("prediction")
        plt.ylabel("ground truth")
        plt.show()
    
    return test_loss

def main():
    dataset = SubjectiveDataset()

    print("example sample")
    print(dataset.__getitem__(0))

    train_samples = int(dataset.__len__() * 0.8)
    eval_samples = dataset.__len__() - train_samples
    train_set, eval_set = random_split(dataset, [train_samples, eval_samples])

    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
    eval_dataloader = DataLoader(eval_set, batch_size=16, shuffle=True)
    example_sample = next(iter(train_dataloader))
    print(example_sample)

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
        val_loss = eval(eval_dataloader, model, loss_fn, device, t%20==0, t==EPOCHS-1)
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
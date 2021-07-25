from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

EPOCHS = 5

def load_projections(data, device):

    # loader uses the transforms function that comes with torchvision
    loader = transforms.Compose([
        transforms.ToTensor()])  

    # Enter the picture address
    # Return tensor variable
    def image_loader(image_name):
        image = Image.open(image_name).convert('RGB').split()[0]
        image = loader(image)
        return image.to(device, torch.float)

    def get_path(model, plane):
        projections_dir = "projections/"
        return f"{projections_dir}{model}_{plane}.png"

    models = data["models"]
    projections = dict()
    for model in models:
        planes = [image_loader(get_path(model, plane)) for plane in ["H", "V", "L"]]
        tensor = torch.cat(planes, 0)
        projections[model] = tensor
    return projections


class SsimDataset(Dataset):
    def __init__(self, data, projections):
        self.dataset = list()
        for sample in data["samples"]:
            pos = sample["pos"]
            pos_ref = sample["pos_ref"]
            model = sample["model"]
            lod_id = data["models"][model].index(sample["lod_name"])
            ssim = sample["ssim"]
            fps = sample["fps"] / 100.0
            self.dataset.append({"input": np.array(pos + pos_ref + [lod_id]), 
                                "projections": projections[model], 
                                "output": np.array([ssim])})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, dropout = 0):
        super(NeuralNetwork, self).__init__()
        hidden_nodes = 256
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(71, hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_nodes, 1)
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Flatten()
        )

    def forward(self, x, projections):
        features = self.cnn(projections)
        nn_inputs = torch.cat((x, features), 1)
        # print(nn_inputs.size())
        output = self.linear_relu_stack(nn_inputs)
        return output

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    train_loss = 0
    for batch, data in enumerate(dataloader):
        input = data["input"].to(device, torch.float)
        projections = data["projections"].to(device, torch.float)
        label = data["output"].to(device, torch.float)

        # Compute prediction error
        pred = model(input, projections)
        loss = loss_fn(pred, label)
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
            input = data["input"].to(device, torch.float)
            projections = data["projections"].to(device, torch.float)
            label = data["output"].to(device, torch.float)
            
            pred = model(input, projections)
            test_loss += loss_fn(pred, label).item()
    test_loss /= size
    print(f"Eval avg loss: {test_loss:>8f} \n")

    if print_example:
        print("example input: " + str(input))
        print("example prediction: " + str(pred))
        print("example label: " + str(label))
    return test_loss

def main():
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    with open("dataset.json", 'r') as f:
        data = json.load(f)

    projections = load_projections(data, device)
    dataset = SsimDataset(data, projections)

    print("example sample")
    print(dataset.__getitem__(0))

    train_samples = int(dataset.__len__() * 0.8)
    eval_samples = dataset.__len__() - train_samples
    train_set, eval_set = random_split(dataset, [train_samples, eval_samples])

    train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(eval_set, batch_size=4, shuffle=True)
    example_sample = next(iter(train_dataloader))
    print(example_sample)

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

    # export model
    dummy_input = (example_sample['input'].to(device, torch.float), 
                   example_sample['projections'].to(device, torch.float))
    torch.onnx.export(model,                       # model being run
                    dummy_input,                   # model dummy input (or a tuple for multiple inputs)
                    "model.onnx",                  # where to save the model (can be a file or file-like object)
                    export_params=True,            # store the trained parameter weights inside the model file
                    opset_version=9,               # the ONNX version to export the model to
                    do_constant_folding=True,      # whether to execute constant folding for optimization
                    input_names = ['input'],  
                    output_names = ['output']
                    )


if __name__ == '__main__':
    main()
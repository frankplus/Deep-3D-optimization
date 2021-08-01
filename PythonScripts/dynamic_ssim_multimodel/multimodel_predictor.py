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

EPOCHS = 50

def load_projections(models, device):
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

    projections = dict()
    for model in models:
        planes = [image_loader(get_path(model, plane)) for plane in ["H", "V", "L"]]
        tensor = torch.cat(planes, 0)
        projections[model] = tensor
    return projections


class SsimDataset(Dataset):
    def __init__(self, samples, models, device):
        projections = load_projections(models, device)
        self.dataset = list()
        for sample in samples:
            pos = sample["pos"]
            pos_ref = sample["pos_ref"]
            model = sample["model"]
            lod_id = models[model].index(sample["lod_name"])
            ssim = sample["ssim"]
            fps = sample["fps"] / 100.0
            self.dataset.append({"input": np.array(pos + pos_ref + [lod_id]), 
                                "projections": projections[model], 
                                "output": np.array([ssim])})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.cnn(x)

class FeedForwardNN(nn.Module):
    def __init__(self, dropout = 0):
        super().__init__()
        hidden_nodes = 256
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(135, hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_nodes, 1)
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class MultimodelNN(nn.Module):
    def __init__(self, cnn, ffn):
        super(MultimodelNN, self).__init__()
        self.cnn = cnn
        self.ffn = ffn

    def forward(self, x, projections):
        features = self.cnn(projections)
        nn_inputs = torch.cat((x, features), 1)
        output = self.ffn(nn_inputs)
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

def export_model(nnmodel, sample):
    cnn = nnmodel.cnn.to("cpu")
    ffn = nnmodel.ffn.to("cpu")
    cnn_dummy_input = sample['projections'].to("cpu", torch.float)
    torch.onnx.export(cnn,
                    cnn_dummy_input,
                    "cnn.onnx",
                    export_params=True,
                    opset_version=9,
                    do_constant_folding=True,
                    input_names = ['input'],
                    output_names = ['output']
                    )

    features = cnn(cnn_dummy_input)
    x = sample['input'].to("cpu", torch.float)
    ffn_dummy_input = torch.cat((x, features), 1)
    torch.onnx.export(ffn,
                    ffn_dummy_input,
                    "ffn.onnx",
                    export_params=True,
                    opset_version=9,
                    do_constant_folding=True,
                    input_names = ['input'],
                    output_names = ['output']
                    )

def train_eval_loop(samples, models, device):
    dataset = SsimDataset(samples, models, device)

    print("example sample")
    print(dataset.__getitem__(0))

    train_samples = int(dataset.__len__() * 0.8)
    eval_samples = dataset.__len__() - train_samples
    train_set, eval_set = random_split(dataset, [train_samples, eval_samples])

    train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(eval_set, batch_size=4, shuffle=True)
    example_sample = next(iter(train_dataloader))

    cnn = ConvNN().to(device)
    ffn = FeedForwardNN().to(device)
    model = MultimodelNN(cnn, ffn).to(device)
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

    # plot results
    plt.plot(train_history, label="train")
    plt.plot(eval_history, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()

    return model, example_sample

def test(nnmodel, samples, models, device):
    print("________ TEST ________")
    dataset = SsimDataset(samples, models, device)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    loss_fn = nn.MSELoss()
    test_loss = eval(dataloader, nnmodel, loss_fn, device, print_example=True)

def main():
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    with open("dataset.json", 'r') as f:
        data = json.load(f)

    trained_model, example_sample = train_eval_loop(data["samples"], data["models"], device)
    test(trained_model, data["test_samples"], data["test_models"], device)
    export_model(trained_model, example_sample)

    plt.show()


if __name__ == '__main__':
    main()
import time

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
import math

EPOCHS = 150

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
            mesh_vertex_count = math.log2(sample["mesh_vertex_count"]) / 30.0
            ssim = sample["ssim"]
            vertex_count = math.log2(sample["vertex_count"]) / 30.0
            self.dataset.append({"input": np.array(pos + pos_ref + [mesh_vertex_count]), 
                                "projections": projections[model], 
                                "output": np.array([ssim, vertex_count])})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Flatten()
        )

    def forward(self, x):
        return self.cnn(x)


class FeedForwardNN(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        hidden_nodes = 256
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(39, hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout * 2),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_nodes, 2),
            nn.Sigmoid()
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
    # print(f"Train avg loss: {train_loss:>8f}")
    return train_loss


def eval(dataloader, model, loss_fn, device, print_example=False):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0

    tot_pred = list()
    tot_lab = list()
    tot_err = list()

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            input = data["input"].to(device, torch.float)
            projections = data["projections"].to(device, torch.float)
            label = data["output"].to(device, torch.float)
            pred = model(input, projections)
            tot_pred.extend(pred)
            tot_lab.extend(label)
            tot_err.extend(abs(pred - label))
            # print(pred.shape)
            test_loss += loss_fn(pred, label).item()
    test_loss /= size
    # print(f"Eval avg loss: {test_loss:>8f} \n")

    avg_err = sum(tot_err) / len(tot_err)
    # print(f"Eval avg SSIM error: {avg_err[0]:>8f}")
    # print(f"Eval avg nlog fvc error: {avg_err[1]:>8f} \n")

    if print_example:
        # print("input: ", str(input))
        print("avg prediction: " + str(sum(tot_pred) / len(tot_pred)))
        print("avg label: " + str(sum(tot_lab) / len(tot_lab)))
        print("avg error (posteriori): " + str(abs(((sum(tot_pred) / len(tot_pred))) - sum(tot_lab) / len(tot_lab))))
        print("avg error (priori): " + str(avg_err) + "\n")
    return test_loss, avg_err


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
                      input_names=['input'],
                      output_names=['output']
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
                      input_names=['input'],
                      output_names=['output']
                      )


def train_eval_loop(samples, models, device, batch=256, val_batch=256):
    # scelgo il gufo per validare
    train_samples = [d for d in samples if d["model"] != 'Owl_high000']
    eval_samples = [d for d in samples if d["model"] == 'Owl_high000']

    train_models = [d for d in models if d != 'Owl_high000']
    eval_models = [d for d in models if d == 'Owl_high000']

    train_set = SsimDataset(samples, models, device)
    eval_set = SsimDataset(samples, models, device)

    train_dataloader = DataLoader(train_set, batch_size=batch, shuffle=True)
    eval_dataloader = DataLoader(eval_set, batch_size=val_batch, shuffle=True)
    example_sample = next(iter(train_dataloader))

    cnn = ConvNN().to(device)
    ffn = FeedForwardNN().to(device)
    model = MultimodelNN(cnn, ffn).to(device)
    print(model)

    # training
    lr = 1e-3
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.6)

    train_history = list()
    eval_history = list()
    val_err_history = list()

    for t in range(EPOCHS):
        # print(f"Epoch {t+1}\n-------------------------------")
        ttt = time.time()
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        val_loss, avg_err = eval(eval_dataloader, model, loss_fn, device, (t + 1) % 10 == 0)
        scheduler.step(train_loss)
        train_history.append(train_loss)
        eval_history.append(val_loss)
        val_err_history.append(avg_err.detach().cpu())
        elapsed = time.time() - ttt
        # print(f"Time to complete the epoch: {elapsed:>8f} \n")
        print(
            f"*EPOCH {t + 1}* train loss: {train_loss:>8f}, val loss: {val_loss:>8f}, SSIM err: {avg_err[0]:>8f}, nlog fvc err: {avg_err[1]:>8f}, time: {elapsed:>8f}")

        # if t%10 == 0:
        #  optimizer = torch.optim.Adam(model.parameters(), lr=lr*0.99)

    # plot results
    plt.plot(train_history, label="train")
    plt.plot(eval_history, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    # plot results
    plt.plot(val_err_history[0], label="SSIM")
    plt.plot(val_err_history[1], label="frame vertex count")
    plt.xlabel("epoch")
    plt.ylabel("avg error")
    plt.legend()
    plt.show()

    return model, example_sample


def test(nnmodel, samples, models, device, batch=256, label="TEST"):
    print("________ " + label + " ________")
    dataset = SsimDataset(samples, models, device)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)
    loss_fn = nn.MSELoss()  # L1Loss()
    test_loss = eval(dataloader, nnmodel, loss_fn, device, print_example=True)


def main():
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    with open("dataset.json", 'r') as f:
        data = json.load(f)

    trained_model, example_sample = train_eval_loop(data["samples"], data["models"], device)

    # overall testing
    test(trained_model, data["test_samples"], data["test_models"], device)

    # test pos ranges
    for i in range(0, 4):
        test(trained_model, [d for d in data["test_samples"] if (
                    math.sqrt(pow(d["pos"][0], 2) + pow(d["pos"][2], 2)) > 0.5 and math.sqrt(
                pow(d["pos_ref"][0], 2) + pow(d["pos_ref"][2], 2)) > 0.5 and d["lod_name"] == 'temple00' + str(i))],
             data["test_models"], device, label="test-ff-lod" + str(i))
        test(trained_model, [d for d in data["test_samples"] if (
                    math.sqrt(pow(d["pos"][0], 2) + pow(d["pos"][2], 2)) > 0.5 and math.sqrt(
                pow(d["pos_ref"][0], 2) + pow(d["pos_ref"][2], 2)) <= 0.5 and d["lod_name"] == 'temple00' + str(i))],
             data["test_models"], device, label="test-fc-lod" + str(i))
        test(trained_model, [d for d in data["test_samples"] if (
                    math.sqrt(pow(d["pos"][0], 2) + pow(d["pos"][2], 2)) <= 0.5 and math.sqrt(
                pow(d["pos_ref"][0], 2) + pow(d["pos_ref"][2], 2)) > 0.5 and d["lod_name"] == 'temple00' + str(i))],
             data["test_models"], device, label="test-cf-lod" + str(i))
        test(trained_model, [d for d in data["test_samples"] if (
                    math.sqrt(pow(d["pos"][0], 2) + pow(d["pos"][2], 2)) <= 0.5 and math.sqrt(
                pow(d["pos_ref"][0], 2) + pow(d["pos_ref"][2], 2)) <= 0.5 and d["lod_name"] == 'temple00' + str(i))],
             data["test_models"], device, label="test-cc-lod" + str(i))

    export_model(trained_model, example_sample)

    plt.show()


if __name__ == '__main__':
    main()
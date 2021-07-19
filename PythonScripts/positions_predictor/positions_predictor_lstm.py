import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

"""
    based on: https://curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/
"""

with open("positions_walk.txt", 'r') as f:
    positions = list()
    for line in f.readlines():
        pos = [float(x) for x in line.split()[:3]]
        positions.append(pos)
    positions = np.array(positions, dtype=np.float32)

train_samples = int(len(positions) * 0.8)
train_data = positions[:train_samples]
val_data = positions[train_samples:]

print(f"train data: {train_data.shape}")
print(f"test data: {val_data.shape}")

scaler = MinMaxScaler()
scaler = scaler.fit(train_data)
train_data = scaler.transform(train_data)
val_data = scaler.transform(val_data)


def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

class SequencesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.dataset = list()
        for i in range(len(data)-seq_length-1):
            x = data[i:(i+seq_length)]
            y = data[i+seq_length]
            self.dataset.append({"X": x, "y": y})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

seq_length = 20
batch_size = 32

train_set = SequencesDataset(train_data, seq_length)
val_set = SequencesDataset(val_data, seq_length)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

class Naive(nn.Module):
    """
        This naive model used just for comparison simply copies 
        the last element of the sequence into output.
    """
    def forward(self, sequences):
        return sequences[:, -1]

    def reset_hidden_state(self, batch_size, device):
        pass


class LSTM(nn.Module):

    def __init__(self, in_features, out_features, n_hidden, seq_len, n_layers=2):
        super(LSTM, self).__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True
        )

        self.linear = nn.Linear(in_features = n_hidden,
                                out_features = out_features)

    def reset_hidden_state(self, batch_size, device):
        self.hidden = (
            torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device),
            torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)
        )

    def forward(self, sequences):
        batch_size, seq_len, _ = sequences.size()
        lstm_out, self.hidden = self.lstm(
            sequences,
            self.hidden
        )
        last_time_step = lstm_out[:, -1]
        y_pred = self.linear(last_time_step)
        return y_pred


def train(dataloader: DataLoader, model, loss_fn, optimizer, print_example=False):
    train_loss = 0
    step = 0
    for batch, data in enumerate(dataloader):
        x_batch = data["X"].to(device)
        y_batch = data["y"].to(device)

        # Compute prediction error
        model.reset_hidden_state(x_batch.size(0), device)
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if print_example and step == 0:
            print("example prediction: " + str(pred))
            print("example label: " + str(y_batch))
        
        step += 1
        
    train_loss /= len(dataloader)
    
    return train_loss

def eval(dataloader: DataLoader, model, loss_fn, print_example=False):
    eval_loss = 0
    step = 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            x_batch = data["X"].to(device)
            y_batch = data["y"].to(device)

            # Compute prediction error
            model.reset_hidden_state(x_batch.size(0), device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            eval_loss += loss.item()

            if print_example and step == 0:
                print("example prediction: " + str(pred[:4]))
                print("example label: " + str(y_batch[:4]))
            
            step += 1
        
    eval_loss /= len(dataloader)
    
    return eval_loss

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = LSTM(
    in_features=3,
    out_features=3,
    n_hidden=128,
    seq_len=seq_length,
    n_layers=2
).to(device)
print(model)

# training
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5)
n_epochs = 100

# calculate loss of naive model for comparison
naive_loss = eval(train_dataloader, Naive(), loss_fn)
print(f"naive model train loss: {naive_loss}")
naive_loss = eval(val_dataloader, Naive(), loss_fn)
print(f"naive model val loss: {naive_loss}")

train_history = list()
eval_history = list()
for t in range(n_epochs):
    print(f"Epoch {t+1} -------------------------------")
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    val_loss = eval(val_dataloader, model, loss_fn, print_example=(t%10==0))
    scheduler.step(val_loss)
    train_history.append(train_loss)
    eval_history.append(val_loss)
    print(f"Train avg loss: {train_loss:>8f}")
    print(f"Eval avg loss: {val_loss:>8f}")


plt.plot(train_history, label="train")
plt.plot(eval_history, label="validation")
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()
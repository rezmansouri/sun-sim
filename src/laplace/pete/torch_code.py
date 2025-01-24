import os
import torch
import h5py as h5
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from os.path import join as path_join


class Model(nn.Module):
    def __init__(self, n_features, dim_hidden, n_layers):
        super(Model, self).__init__()
        self.n_features = n_features
        self.dim_hidden = dim_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            n_features, hidden_size=dim_hidden, num_layers=n_layers, batch_first=True
        )
        self.dense = nn.Linear(in_features=dim_hidden, out_features=n_features)

    def forward(self, x, h_0=None, c_0=None):
        if h_0 is None:
            h_0 = torch.zeros(self.n_layers, x.shape[0], self.dim_hidden)
            c_0 = torch.zeros(self.n_layers, x.shape[0], self.dim_hidden)
        x, (h, c) = self.lstm(x, (h_0, c_0))
        return self.dense(x), (h, c)


def main():

    # read files
    train_path = "/Users/reza/Career/DMLab/SURROGATE/data/pete_sample/train"
    val_path = "/Users/reza/Career/DMLab/SURROGATE/data/pete_sample/test"
    train_arrays = []
    for h5_name in os.listdir(train_path):
        with h5.File(path_join(train_path, h5_name)) as h5_file:
            train_arrays.append(h5_file["Data"][:])
    val_arrays = []
    for h5_name in os.listdir(val_path):
        with h5.File(path_join(val_path, h5_name)) as h5_file:
            val_arrays.append(h5_file["Data"][:])

    # normalize
    for i in range(len(train_arrays)):
        train_arrays[i] /= np.max(train_arrays[i])
    for i in range(len(val_arrays)):
        val_arrays[i] /= np.max(val_arrays[i])

    # transpose
    for i in range(len(train_arrays)):
        train_arrays[i] = np.transpose(train_arrays[i], (2, 0, 1))
    for i in range(len(val_arrays)):
        val_arrays[i] = np.transpose(val_arrays[i], (2, 0, 1))

    # accumulate
    train_set = np.concatenate(train_arrays, axis=0)
    val_set = np.concatenate(val_arrays, axis=0)

    # X, y
    train_X = train_set[:-1, :, :]
    train_y = train_set[1:, :, :]
    val_X = val_set[:-1, :, :]
    val_y = val_set[1:, :, :]

    # tensorize
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_X = torch.tensor(val_X, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)

    # print(train_X.shape)
    # print(train_y.shape)

    # print(val_X.shape)
    # print(val_y.shape)

    # shapes
    dim_k_train, dim_i, dim_j = train_X.shape
    dim_k_val, dim_i, dim_j = val_X.shape

    # model
    hidden_size = 64
    num_layers = 1
    model = Model(dim_j, hidden_size, num_layers)

    # criterion
    loss_fn = nn.MSELoss()

    # optim
    optimizer = torch.optim.Adam(model.parameters())

    # batch_size
    train_batch_size = 64
    val_batch_size = 139

    # train
    best_val_loss = torch.inf
    best_state = None
    best_epoch = -1
    train_loss, val_loss = [], []
    for epoch in range(1, 101):
        print("epoch:", epoch, end=" ")
        t_loss = []
        model.train()
        for i in range(0, dim_k_train, train_batch_size):

            x = train_X[i : i + train_batch_size, :, :]
            y = train_y[i : i + train_batch_size, :, :]

            yhat, _ = model(x)
            loss = loss_fn(yhat, y)
            t_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t_loss = np.mean(t_loss)
        train_loss.append(t_loss)
        print("training loss:", t_loss, end=" ")
        v_loss = []
        model.eval()
        for i in range(0, dim_k_val, val_batch_size):
            x = val_X[i : i + val_batch_size, :1, :]
            y = val_y[i : i + val_batch_size, :, :]
            yhat = torch.zeros_like(y, dtype=y.dtype)
            h = torch.zeros(num_layers, val_batch_size, hidden_size)
            c = torch.zeros(num_layers, val_batch_size, hidden_size)
            for t in range(dim_i):
                with torch.no_grad():
                    yhat_t, (h, c) = model(x, h, c)
                    x = yhat_t
                    yhat[i : i + val_batch_size, t : t + 1, :] = yhat_t
            loss = loss_fn(yhat, y)
            v_loss.append(loss.item())
        v_loss = np.mean(v_loss)
        if v_loss < best_val_loss:
            best_epoch = epoch
            best_state = model.state_dict()
            best_val_loss = v_loss
        val_loss.append(v_loss)
        print("validation loss:", v_loss)

    np.save("train_loss.npy", np.array(train_loss))
    np.save("val_loss.npy", np.array(val_loss))
    torch.save(best_state, f"{best_epoch}.pth")


if __name__ == "__main__":
    main()

import sys
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from model import ConvLSTM
import matplotlib.pyplot as plt
from torch.optim.adam import Adam
from utils import TrainDataset, TestDataset
from torch.utils.data.dataloader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # train_path = '/Users/reza/Career/DMLab/SURROGATE/data/pete_sample/train'
    # val_path = '/Users/reza/Career/DMLab/SURROGATE/data/pete_sample/test'
    # batch_size = 2
    # num_layers = 1
    # hidden_dim = 4
    # n_epochs = 10
    train_path, val_path, batch_size, num_layers, hidden_dim, n_epochs = sys.argv[1:]
    batch_size, num_layers, hidden_dim, n_epochs = [
        int(i) for i in [batch_size, num_layers, hidden_dim, n_epochs]
    ]
    train_dataset = TrainDataset(train_path)
    train_min, train_max = train_dataset.get_min_max()
    test_dataset = TestDataset(val_path, train_min, train_max)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = ConvLSTM(
        input_dim=1,
        hidden_dim=hidden_dim,
        kernel_size=(3, 3),
        num_layers=num_layers,
        batch_first=True,
        bias=True,
        return_all_layers=False,
        output_dim=1
    )
    optimizer = Adam(model.parameters())
    loss_fn = nn.MSELoss()

    best_val_loss = torch.inf
    best_state = None
    best_epoch = -1
    train_loss, val_loss = [], []

    seq_len = train_dataset[0][0].shape[0]

    for epoch in range(1, n_epochs + 1):
        print("epoch:", f"{epoch}/{n_epochs}", end="\t")
        t_loss = []
        model.train()
        for x, y in train_loader:

            yhat, _ = model(x)
            loss = loss_fn(yhat, y)
            t_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t_loss = np.mean(t_loss)
        train_loss.append(t_loss)
        print("training loss:", t_loss, end='\t')
        v_loss = []
        model.eval()
        for x, y in test_loader:
            with torch.no_grad():
                yhat, _ = model(x, teacher_forcing=False, seq_len=seq_len)
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

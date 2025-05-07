import os
import sys
import numpy as np
import torch
import json
from neuralop import LpLoss
from neuralop import Trainer
from neuralop.models import SFNO
from trainer import train_cv, train
from utils import SphericalNODataset, get_cr_dirs
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    data_path, batch_size, n_epochs = sys.argv[1:]
    batch_size, n_epochs = int(batch_size), int(n_epochs)

    cr_dirs = get_cr_dirs(data_path)
    cr_train, cr_val = train_test_split(cr_dirs, test_size=0.2, random_state=42)

    hyperparams = {
        "hidden_channels": [8],
        "n_modes": [8],
        "projection_channel_ratio": [1],
        "factorization": ["tt"],
        "lr": [8e-4],
        "weight_decay": [0.0],
    }

    loss_fn = LpLoss(d=2, p=2, reduction="sum")

    results = train_cv(
        data_path,
        cr_train,
        hyperparams,
        5,
        n_epochs,
        batch_size,
        loss_fn,
        device=device,
    )

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("Training completed. Results saved to results.json.")


if __name__ == "__main__":
    main()

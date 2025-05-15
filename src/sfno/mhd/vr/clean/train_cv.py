import sys
import torch
import json
from trainer import train_cv
from utils import get_cr_dirs, AreaWeightedLpLoss

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    data_path, batch_size, n_epochs, area_weighted = sys.argv[1:]
    batch_size, n_epochs, area_weighted = (
        int(batch_size),
        int(n_epochs),
        bool(area_weighted),
    )

    cr_dirs = get_cr_dirs(data_path)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train = cr_dirs[:split_ix]

    hyperparams = {
        "hidden_channels": [8],
        "n_modes": [8],
        "projection_channel_ratio": [1],
        "factorization": ["tt"],
        "lr": [8e-4],
        "weight_decay": [0.0],
    }

    loss_fn = AreaWeightedLpLoss(d=2, p=2, reduction="sum", area_weighted=area_weighted)

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

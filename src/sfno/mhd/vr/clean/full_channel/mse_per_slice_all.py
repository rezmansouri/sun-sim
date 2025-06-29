import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from neuralop.models import SFNO
from utils import get_cr_dirs, SphericalNODataset
from torch.utils.data import DataLoader
from metrics import mse_score_per_slice, mse_score_per_slice_masked, sobel_edge_map

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    data_path, result_path = sys.argv[1:]
    cr_dirs = get_cr_dirs(data_path)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train, cr_val = cr_dirs[:split_ix], cr_dirs[split_ix:]

    train_dataset = SphericalNODataset(data_path, cr_train, scale_up=1)
    val_dataset = SphericalNODataset(
        data_path,
        cr_val,
        v_min=train_dataset.v_min,
        v_max=train_dataset.v_max,
        scale_up=1,
    )

    cfg_path = os.path.join(result_path, "cfg.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        pin_memory=True,
    )

    model = SFNO(
        n_modes=(110, 128),
        in_channels=1,
        out_channels=139,
        hidden_channels=cfg["hidden_channels"],
        projection_channel_ratio=2,
        factorization="dense",
        n_layers=cfg["n_layers"],
    ).to(device)
    state = torch.load(
        os.path.join(result_path, "model.pt"), map_location=device, weights_only=False
    )
    model.load_state_dict(state)

    mses = []
    masked_mses = []

    model.eval()

    with torch.no_grad():
        for instance in tqdm(val_loader, leave=False):

            x, cube = instance["x"].to(device), instance["y"].to(device)
            yhats = model(x)  # shape (B, T-1, H, W)

            yhats = yhats * (cfg["v_max"] - cfg["v_min"]) + cfg["v_min"]
            cube = cube * (cfg["v_max"] - cfg["v_min"]) + cfg["v_min"]

            yhats *= 481.3711
            cube *= 481.3711

            mask = sobel_edge_map(cube)
            mses_batch = mse_score_per_slice(cube, yhats)
            mses_masked_batch = mse_score_per_slice_masked(cube, yhats, mask)

            mses.extend(mses_batch.detach().cpu().numpy().tolist())
            masked_mses.extend(mses_masked_batch.detach().cpu().numpy().tolist())

    mses = np.array(mses)
    mse_df_dict = {f"slice_{i+1}": mses[:, i] for i in range(mses.shape[1])}
    df = pd.DataFrame(mse_df_dict)
    df.to_csv(os.path.join(result_path, "mse_per_slice_all.csv"), index=False)
    
    masked_mses = np.array(masked_mses)
    masked_mse_df_dict = {f"slice_{i+1}": masked_mses[:, i] for i in range(masked_mses.shape[1])}
    df = pd.DataFrame(masked_mse_df_dict)
    df.to_csv(os.path.join(result_path, "masked_mse_per_slice_all.csv"), index=False)


    print("Done!")


if __name__ == "__main__":
    main()

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
from metrics import psnr_score_per_sample, psnr_score_per_sample_masked, sobel_edge_map

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    data_path, result_path = sys.argv[1:]
    cr_dirs = get_cr_dirs(data_path)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train, cr_val = cr_dirs[:split_ix], cr_dirs[split_ix:]

    train_dataset = SphericalNODataset(data_path, cr_train)
    val_dataset = SphericalNODataset(
        data_path,
        cr_val,
        v_min=train_dataset.v_min,
        v_max=train_dataset.v_max,
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

    buffer = cfg["buffer"]

    model = SFNO(
        n_modes=(110, 128),
        in_channels=1,
        out_channels=buffer,
        hidden_channels=cfg["hidden_channels"],
        projection_channel_ratio=2,
        factorization="dense",
        n_layers=cfg["n_layers"],
    ).to(device)
    state = torch.load(
        os.path.join(result_path, "model.pt"), map_location=device, weights_only=False
    )
    model.load_state_dict(state)

    psnrs = []
    masked_psnrs = []
    final_paths = []

    model.eval()

    with torch.no_grad():
        for cube, paths in tqdm(val_loader, leave=False):

            t = cube.shape[1]  # 140
            x = cube[:, 0:1, :, :].to(device)  # initial input
            pred = []

            # Step through time autoregressively
            for i in trange(0, t - 1, buffer, leave=False):
                y_start = i + 1
                y_end = min(i + buffer + 1, t)

                y = cube[:, y_start:y_end, :, :].to(
                    device
                )  # target: ground truth next slices

                # print(y_start, y_end, y.shape)
                pred_buf = model(x)[
                    :, : y.shape[1], :, :
                ]  # model predicts next <buffer> steps

                pred.append(pred_buf)

                # next input is last predicted frame (1-step only)
                x = pred_buf[:, -1:, :, :].detach()

            yhats = torch.cat(pred, dim=1)  # shape (B, T-1, H, W)
            cube = cube[:, 1:, :, :].to(device)

            yhats = yhats * (cfg["v_max"] - cfg["v_min"]) + cfg["v_min"]
            cube = cube * (cfg["v_max"] - cfg["v_min"]) + cfg["v_min"]

            yhats *= 481.3711
            cube *= 481.3711

            mask = sobel_edge_map(cube)
            psnrs_batch = psnr_score_per_sample(cube, yhats)
            masked_psnrs_batch = psnr_score_per_sample_masked(cube, yhats, mask)

            psnrs.extend(psnrs_batch.detach().cpu().numpy().tolist())
            masked_psnrs.extend(masked_psnrs_batch.detach().cpu().numpy().tolist())

            final_paths.extend(paths)

    psnr = np.sum(psnrs) / len(val_dataset)
    masked_psnr = np.sum(masked_psnrs) / len(val_dataset)

    with open(
        os.path.join(result_path, "evaluation_psnr.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "psnr": float(psnr),
                "masked_psnr": float(masked_psnr),
            },
            f,
            indent=4,
        )

    df = pd.DataFrame({"psnr": psnrs, "masked_psnr": masked_psnrs})
    df.to_csv(os.path.join(result_path, "evaluation_psnr.csv"), index=False)

    print("Done!")


if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import torch
import json
from trainer import train
from neuralop.models import CODANO
from neuralop.losses import LpLoss
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.spherical_convolution import SphericalConv
import torch.nn.functional as F
from utils import CODANODataset, get_cr_dirs

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    (
        data_path,
        batch_size,
        n_epochs,
        hidden_channels,
        n_layers,
        conv,
        physics_loss,
        cr_size,
    ) = sys.argv[1:]
    (
        batch_size,
        n_epochs,
        hidden_channels,
        n_layers,
        physics_loss,
        cr_size,
    ) = (
        int(batch_size),
        int(n_epochs),
        int(hidden_channels),
        int(n_layers),
        int(physics_loss) == 1,
        int(cr_size),
    )

    # if pos_embedding == 'none':
    #     pos_embedding = None

    cr_dirs = get_cr_dirs(data_path)
    if cr_size != -1:
        cr_dirs = cr_dirs[:cr_size]
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train, cr_val = cr_dirs[:split_ix], cr_dirs[split_ix:]

    train_dataset = CODANODataset(data_path, cr_train)
    val_dataset = CODANODataset(
        data_path,
        cr_val,
        br_min=train_dataset.br_min,
        br_max=train_dataset.br_max,
        bt_min=train_dataset.bt_min,
        bt_max=train_dataset.bt_max,
        bp_max=train_dataset.bp_max,
        bp_min=train_dataset.bp_min,
        jr_min=train_dataset.jr_min,
        jr_max=train_dataset.jr_max,
        jt_min=train_dataset.jt_min,
        jt_max=train_dataset.jt_max,
        jp_max=train_dataset.jp_max,
        jp_min=train_dataset.jp_min,
    )

    if conv == "sfno":
        conv_module = SphericalConv
    elif conv == "fno":
        conv_module = SpectralConv
    else:
        raise ValueError("conv_module must be either 'sfno' or 'fno'")

    loss_fn = LpLoss(d=2, p=2)

    out_path = f"jb_convolution_{conv}_n_layers-{n_layers}_hidden_channels-{hidden_channels}_physics-{physics_loss}"
    os.makedirs(
        out_path,
        exist_ok=True,
    )

    cfg = {
        "num_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": 8e-4,
        "train_files": cr_train,
        "val_files": cr_val,
        **train_dataset.get_min_max(),
        "hidden_channels": hidden_channels,
        "n_layers": n_layers,
        "physics_loss": physics_loss,
    }
    with open(os.path.join(out_path, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)

    # if pos_embedding == 'pt':
    #     in_channels = 4
    # elif pos_embedding == 'ptr':
    #     raise ValueError('radii embedding is the same in full channel and is not supported here')
    # elif pos_embedding is None:
    #     in_channels = 1
    # else:
    #     raise ValueError('wrong pos embedding')

    model = CODANO(
        n_layers=n_layers,
        n_modes=[[110, 128] * n_layers],
        output_variable_codimension=139,
        lifting_channels=64,
        hidden_variable_codimension=hidden_channels,
        projection_channels=64,
        use_positional_encoding=False,
        positional_encoding_dim=None,
        positional_encoding_modes=None,
        static_channel_dim=0,
        variable_ids=None,
        per_layer_scaling_factors=None,
        n_heads=None,
        attention_scaling_factors=None,
        conv_module=conv_module,
        nonlinear_attention=False,
        non_linearity=F.gelu,
        attention_token_dim=1,
        per_channel_attention=False,
        enable_cls_token=False,
        use_horizontal_skip_connection=False,
        horizontal_skips_map=None,
        domain_padding=0, # Default: 0.25
    ).to(device)

    # sfno = SFNO(
    #     n_modes=(110, 128),
    #     in_channels=6 * encoder_hidden_channels,
    #     out_channels=6 * 139,
    #     hidden_channels=hidden_channels,
    #     factorization="dense",
    #     projection_channel_ratio=2,
    #     n_layers=n_layers,
    #     positional_embedding=None,  # "grid is default"
    # ).to(device)

    (
        train_losses,
        val_losses,
        train_physics_loss,
        val_physics_loss,
        # train_mse,
        # val_mse,
        # train_mse_masked,
        # val_mse_masked,
        # train_msssim,
        # val_msssim,
        # train_acc,
        # val_acc,
        # train_psnr,
        # val_psnr,
        best_epoch,
        best_state_dict,
    ) = train(
        model,
        train_dataset,
        val_dataset,
        n_epochs=n_epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        device=device,
        lr=8e-4,
        weight_decay=0.0,
        physics_informed=physics_loss,
    )

    torch.save(best_state_dict, os.path.join(out_path, "model.pt"))
    with open(
        os.path.join(out_path, f"best_epoch-{best_epoch}.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(f"best_epoch: {best_epoch}")
    np.save(os.path.join(out_path, "train_losses.npy"), train_losses)
    np.save(os.path.join(out_path, "val_losses.npy"), val_losses)
    np.save(os.path.join(out_path, "train_physics_loss.npy"), train_physics_loss)
    np.save(os.path.join(out_path, "val_physics_loss.npy"), val_physics_loss)
    # np.save(os.path.join(out_path, "train_rmse.npy"), train_rmse)
    # np.save(os.path.join(out_path, "val_rmse.npy"), val_rmse)
    # np.save(os.path.join(out_path, "train_nnse.npy"), train_nnse)
    # np.save(os.path.join(out_path, "val_nnse.npy"), val_nnse)
    # np.save(os.path.join(out_path, "train_msssim.npy"), train_msssim)
    # np.save(os.path.join(out_path, "val_msssim.npy"), val_msssim)
    # np.save(os.path.join(out_path, "train_acc.npy"), train_acc)
    # np.save(os.path.join(out_path, "val_acc.npy"), val_acc)
    # np.save(os.path.join(out_path, "train_psnr.npy"), train_psnr)
    # np.save(os.path.join(out_path, "val_psnr.npy"), val_psnr)
    print("Training completed.")


if __name__ == "__main__":
    main()

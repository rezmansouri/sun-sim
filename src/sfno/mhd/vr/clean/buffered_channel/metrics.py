import torch


def rmse_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute Root Mean Squared Error (RMSE) between y_true and y_pred.

    Args:
        y_true (torch.Tensor): Ground truth tensor
        y_pred (torch.Tensor): Predicted tensor

    Returns:
        float: RMSE value
    """
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    return float(rmse.item())


def nnse_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    climatology: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    Compute the Normalized Nash–Sutcliffe Efficiency (NNSE) score.

    Args:
        y_true (torch.Tensor): Ground truth, shape (B, D, H, W)
        y_pred (torch.Tensor): Prediction, shape (B, D, H, W)
        climatology (torch.Tensor): Climatology mean field, shape (D, H, W)
        eps (float): Small number to avoid divide-by-zero

    Returns:
        float: NNSE score (higher is better, max = 1)
    """
    assert (
        y_true.shape == y_pred.shape
    ), f"y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape"
    assert (
        y_true.shape[1:] == climatology.shape
    ), f"climatology {climatology.shape} must match spatial shape {y_true.shape[1:]}"

    # Expand climatology to match batch size
    clim = climatology.unsqueeze(0).expand_as(y_true)

    # Compute NSE
    numerator = torch.sum((y_true - y_pred) ** 2)
    denominator = torch.sum((y_true - clim) ** 2).clamp(min=eps)
    nse = 1 - numerator / denominator

    # Compute NNSE
    nnse = 1 / (2 - nse)
    return float(nnse.item())


def acc_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    climatology: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    Compute Anomaly Correlation Coefficient (ACC).

    Args:
        y_true (torch.Tensor): Ground truth, shape (B, D, H, W)
        y_pred (torch.Tensor): Model prediction, shape (B, D, H, W)
        climatology (torch.Tensor): Climatology mean field, shape (D, H, W)
        eps (float): Small number to avoid division by zero

    Returns:
        float: ACC score
    """
    assert (
        y_true.shape == y_pred.shape
    ), f"y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape"
    assert (
        y_true.shape[1:] == climatology.shape
    ), f"climatology {climatology.shape} must match spatial shape {y_true.shape[1:]}"

    clim = climatology.unsqueeze(0).expand_as(y_true)

    # Compute anomalies
    y_true_anom = y_true - clim
    y_pred_anom = y_pred - clim

    # Numerator: dot product of anomalies
    numerator = torch.sum(y_true_anom * y_pred_anom)

    # Denominator: product of norms
    denom = torch.norm(y_true_anom) * torch.norm(y_pred_anom)
    denom = denom.clamp(min=eps)

    acc = numerator / denom
    return float(acc.item())


def psnr_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-10,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between y_true and y_pred.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (B, ...)
        y_pred (torch.Tensor): Predicted tensor of same shape
        data_range (float): Max value range of the data (1.0 if normalized, 255.0 for 8-bit images)
        eps (float): Small value to avoid division by zero

    Returns:
        float: Mean PSNR over the batch
    """
    assert (
        y_true.shape == y_pred.shape
    ), f"y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape"
    assert (
        y_true.dtype == y_pred.dtype
    ), f"Input dtypes must match {y_true.dtype} vs {y_pred.dtype}"

    # Compute MSE per sample
    mse = torch.mean((y_true - y_pred) ** 2, dim=0)
    max_ = torch.max(y_true, dim=0)[0]
    psnr = 10 * torch.log10((max_**2) / (mse + eps))
    return float(psnr.mean().item())


def mssim_score(mssim_module, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    eps = 1e-6
    B = y_true.shape[0]

    # Reshape: treat (138,111,128) as 3D volume with 1 channel
    y_true = y_true.unsqueeze(1)  # (B, 1, 138, 111, 128)
    y_pred = y_pred.unsqueeze(1)

    # Min-max normalization
    y_true_flat = y_true.reshape(B, -1)
    y_pred_flat = y_pred.reshape(B, -1)

    min_vals_true = y_true_flat.min(dim=1, keepdim=True).values.view(B, 1, 1, 1, 1)
    max_vals_true = y_true_flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1, 1)

    min_vals_pred = y_pred_flat.min(dim=1, keepdim=True).values.view(B, 1, 1, 1, 1)
    max_vals_pred = y_pred_flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1, 1)

    range_vals_true = (max_vals_true - min_vals_true).clamp(min=eps)

    range_vals_pred = (max_vals_pred - min_vals_pred).clamp(min=eps)

    y_true_norm = (y_true - min_vals_true) / range_vals_true
    y_pred_norm = (y_pred - min_vals_pred) / range_vals_pred

    score = mssim_module(y_true_norm, y_pred_norm.to(y_true_norm.dtype))
    return float(score.item())

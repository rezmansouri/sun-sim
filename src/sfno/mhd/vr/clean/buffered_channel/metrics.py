import torch
import torch.nn.functional as F


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


def sobel_edge_map(batch_cube: torch.Tensor) -> torch.Tensor:
    """
    Apply Sobel edge detection on each (H, W) frame in a tensor of shape (B, C, H, W).
    Returns a binary mask of shape (B, C, H, W) where edge pixels are 1.
    """
    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=batch_cube.dtype,
        device=batch_cube.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=batch_cube.dtype,
        device=batch_cube.device,
    ).view(1, 1, 3, 3)

    # Pad the images to preserve size
    padded = F.pad(
        batch_cube, (1, 1, 1, 1), mode="replicate"
    )  # shape: (B, C, H+2, W+2)

    # Apply Sobel filters per channel
    dx = F.conv2d(
        padded, sobel_x.repeat(batch_cube.shape[1], 1, 1, 1), groups=batch_cube.shape[1]
    )
    dy = F.conv2d(
        padded, sobel_y.repeat(batch_cube.shape[1], 1, 1, 1), groups=batch_cube.shape[1]
    )

    # Compute gradient magnitude
    grad_mag = torch.sqrt(dx**2 + dy**2)

    # Normalize and threshold to get binary edge map
    edge_mask = (grad_mag > grad_mag.mean(dim=(-2, -1), keepdim=True)).to(
        batch_cube.dtype
    )

    return edge_mask


def mse_score_masked(
    y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor
) -> float:
    """
    Compute Root Mean Squared Error (RMSE) between y_true and y_pred,
    considering only the elements where the mask is True (or 1).

    Args:
        y_true (torch.Tensor): Ground truth tensor
        y_pred (torch.Tensor): Predicted tensor
        mask (torch.Tensor): Binary mask tensor of the same shape as y_true and y_pred.
                             RMSE is computed only where mask is 1.

    Returns:
        float: RMSE value, or 0.0 if the mask is all zeros.
    """
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"
    assert y_true.shape == mask.shape, "Shape of mask must match y_true and y_pred"
    assert (
        mask.dtype == torch.bool or mask.dtype == y_true.dtype
    ), "Mask dtype should be bool or match y_true dtype"

    # Ensure mask is boolean for indexing if it's not already
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # Apply the mask
    masked_diff = (y_true - y_pred)[mask]

    if masked_diff.numel() == 0:
        # Handle cases where the mask is all zeros to avoid division by zero
        return 0.0

    # Compute MSE on the masked elements
    mse = torch.mean(masked_diff**2)
    return float(mse.item())


def mse_score_per_sample(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE) for each sample in a batch
    between y_true and y_pred.

    Args:
        y_true (torch.Tensor): Ground truth tensor, shape (B, ...)
                               where B is the batch size.
        y_pred (torch.Tensor): Predicted tensor, shape (B, ...)
                               must match y_true.shape.

    Returns:
        torch.Tensor: A 1D tensor of RMSE values of shape (B,),
                      containing the RMSE for each sample in the batch.
    """
    assert (
        y_true.shape == y_pred.shape
    ), f"Shapes of y_true ({y_true.shape}) and y_pred ({y_pred.shape}) must match"
    assert (
        y_true.ndim > 0
    ), "Input tensors must have at least one dimension (batch size)."

    # Calculate squared error
    squared_error = (y_true - y_pred) ** 2

    # Define dimensions to reduce over (all dimensions except the batch dimension 0)
    if y_true.ndim == 1:  # Batch of scalars, e.g. shape (B,)
        # For a batch of scalars, squared_error is already the squared error per sample.
        # MSE per sample is just the squared_error itself.
        mse_per_sample = squared_error
    else:  # Batch of tensors, e.g. shape (B, C, H, W)
        reduce_dims = tuple(range(1, y_true.ndim))
        mse_per_sample = torch.mean(squared_error, dim=reduce_dims)


    return mse_per_sample


def mse_score_per_sample_masked(
    y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE) for each sample in a batch
    between y_true and y_pred, considering only masked regions.

    Args:
        y_true (torch.Tensor): Ground truth tensor, shape (B, ...)
                               where B is the batch size.
        y_pred (torch.Tensor): Predicted tensor, shape (B, ...)
                               must match y_true.shape.
        mask (torch.Tensor): Boolean mask tensor, shape (B, ...)
                             must match y_true.shape. RMSE is computed
                             only where mask is True.

    Returns:
        torch.Tensor: A 1D tensor of RMSE values of shape (B,),
                      containing the RMSE for each sample in the batch.
                      If a sample has no elements selected by its mask,
                      its RMSE will be 0.0.
    """
    assert (
        y_true.shape == y_pred.shape
    ), f"Shapes of y_true ({y_true.shape}) and y_pred ({y_pred.shape}) must match"
    assert (
        y_true.shape == mask.shape
    ), f"Shape of mask ({mask.shape}) must match y_true ({y_true.shape})"
    assert mask.dtype == torch.bool, "Mask tensor must be of boolean type."
    assert (
        y_true.ndim > 0
    ), "Input tensors must have at least one dimension (batch size)."

    # Calculate squared error
    squared_error = (y_true - y_pred) ** 2

    # Apply the mask to the squared error.
    # Elements where mask is False will contribute 0 to the sum of squared errors.
    # We multiply by mask converted to float to ensure correct summation.
    masked_squared_error = squared_error * mask.to(squared_error.dtype)

    # Define dimensions to reduce over (all dimensions except the batch dimension 0)
    if y_true.ndim == 1:  # Batch of scalars, e.g. shape (B,)
        # For a batch of scalars, sum over no dimensions is the element itself.
        sum_masked_se_per_sample = masked_squared_error
        # Count of masked elements is 1 if mask is True, 0 if False.
        num_masked_elements_per_sample = mask.to(squared_error.dtype)
    else:  # Batch of tensors, e.g. shape (B, C, H, W)
        reduce_dims = tuple(range(1, y_true.ndim))
        sum_masked_se_per_sample = torch.sum(masked_squared_error, dim=reduce_dims)
        num_masked_elements_per_sample = torch.sum(
            mask.to(squared_error.dtype), dim=reduce_dims
        )

    # Initialize MSE per sample as zeros.
    # This ensures that if num_masked_elements_per_sample is 0, MSE remains 0.
    mse_per_sample = torch.zeros_like(sum_masked_se_per_sample)

    # Create a boolean mask for samples that have at least one masked element
    has_masked_elements = num_masked_elements_per_sample > 0

    # Calculate MSE only for samples that have masked elements to avoid division by zero
    mse_per_sample[has_masked_elements] = (
        sum_masked_se_per_sample[has_masked_elements]
        / num_masked_elements_per_sample[has_masked_elements]
    )

    return mse_per_sample

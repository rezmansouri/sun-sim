"""Computing metrics as lazy dataset (with dask DAG)."""

import numpy as np
import xarray as xr
from xverif.utils.timing import print_elapsed_time

EPS = 1e-6


def spearmanr(a_rank, b_rank, axis=-1, assume_noties=False):
    """Compute Spearman correlation coefficient."""
    n = a_rank.shape[axis]
    if assume_noties:
        return 1 - 6 * ((a_rank - b_rank) ** 2).sum(axis=axis) / (n * (n**2 - 1))
    else:
        a_rank_mean = (1 + n) / 2
        b_rank_mean = (1 + n) / 2
        a_rank_std = np.sqrt(((n - 1) * (n + 1)) / 12)
        b_rank_std = np.sqrt(((b_rank.shape[-1] - 1) * (b_rank.shape[-1] + 1)) / 12)
        out = ((a_rank - a_rank_mean) * (b_rank - b_rank_mean)).mean(axis=-1) / (
            a_rank_std * b_rank_std
        )
        return out


def rank(x: np.ndarray, axis: int = -1):
    """Rank an n-dimensional array along a specified axis."""
    in_shape = x.shape
    tmp = np.argsort(x, axis=-1).reshape(-1, in_shape[axis])
    rank = np.empty_like(tmp)
    np.put_along_axis(rank, tmp, np.arange(1, in_shape[axis] + 1), axis=axis)
    del tmp
    rank = rank.reshape(in_shape)
    return rank


def _get_metrics_lazy(
    pred: xr.DataArray, obs: xr.DataArray, dim: str | list[str], **kwargs
):
    """Deterministic metrics for continuous predictions forecasts.

    This function expects pred and obs to be 1D vector of same size
    """
    p: xr.Variable = pred.variable
    o: xr.Variable = obs.variable

    dim = [dim] if isinstance(dim, str) else dim
    # n = int(np.product([p.sizes[d] for d in list(dim)]))
    out_coords = {d: v for d, v in pred.coords.items() if d != dim}
    out_indexes = {d: v for d, v in pred.indexes.items() if d != dim}

    def _dataarray(x: xr.Variable, name: str) -> xr.DataArray:
        return xr.DataArray(
            x, coords=out_coords, indexes=out_indexes, name=name, fastpath=True
        )

    error = p - o
    error_squared = error**2
    # error_perc = error / (obs + EPS)

    pred_mean = p.mean(dim)
    obs_mean = o.mean(dim)
    error_mean = error.mean(dim)
    pred_std = p.std(dim)
    obs_std = o.std(dim)
    pred_CoV = pred_std / (pred_mean + EPS)
    obs_CoV = obs_std / (obs_mean + EPS)

    covariance = ((p - pred_mean) * (o - obs_mean)).mean(dim=dim)
    pearson_r = covariance / (pred_std * obs_std)
    obs_rank = xr.apply_ufunc(rank, o, dask="parallelized")
    pred_rank = xr.apply_ufunc(rank, p, dask="parallelized")
    spearman_r = spearmanr(
        pred_rank, obs_rank, axis=p.dims.index(dim), assume_noties=True
    )

    results = [
        _dataarray(pred_CoV, name="pred_coefficient_of_variation"),
        _dataarray(obs_CoV, name="obs_coefficient_of_variation"),
        _dataarray(pred_mean - obs_mean, name="bias"),
        _dataarray(pred_mean / obs_mean, name="multiplicative_bias"),
        _dataarray(abs(error).mean(dim), name="mean_absolute_error"),
        _dataarray(error_mean, name="mean_error"),
        _dataarray((es := error_squared.mean(dim)), name="mean_squared_error"),
        _dataarray(np.sqrt(es), name="root_mean_squared_error"),
        _dataarray(pearson_r, "pearson_r"),
        _dataarray(spearman_r, "spearman_r"),
    ]

    return xr.merge(results)


@print_elapsed_time(task="deterministic continuous")
def _get_metrics(
    pred,
    obs,
    sample_dims,
    metrics=None,
    compute=True,
    **kwargs,
):
    """Compute deterministic continuous metrics.

    dims must be a tuple, unique values
    """
    metrics_ds = _get_metrics_lazy(pred, obs, sample_dims)
    metrics_ds = metrics_ds[metrics] if metrics else metrics_ds
    metrics_ds = metrics_ds.compute() if compute else metrics_ds

    return metrics_ds


__all__ = ["_get_metrics_lazy", "_get_metrics"]

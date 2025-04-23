#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:59:26 2023.

@author: ghiggi
"""
import dask.array
import numpy as np
import xarray as xr

#### rankdata
# try:
#     from bottleneck import nanrankdata as rankdata
# except ImportError:
#     from scipy.stats import rankdata

### Ranks benchmarks
# import bottleneck
# import scipy.stats
# arr = np.arange(0, 100_000_000).reshape(10_000,10_000)

# %timeit arr.argsort(axis=1).argsort(axis=1) + 1
# %timeit rankdata(arr, axis=1)
# %timeit bottleneck.rankdata(arr, axis=1)
# %timeit scipy.stats.rankdata(arr, axis=1)

# x = np.array([np.nan, np.nan, 5, 4])
# x.argsort().argsort() + 1
# rankdata(x)
# bottleneck.rankdata(x)
# bottleneck.nanrankdata(x)
# scipy.stats.rankdata(x)

####---------------------------------------------------------------------------.
#### pvalues
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L319
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L374
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L460
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L503


####---------------------------------------------------------------------------.


def _np_rankdata(x: np.ndarray, axis: int = -1, out_dtype="int32"):
    """Rank data using numpy.

    Avoids using argsort 2 times.
    """
    in_shape = x.shape
    tmp = np.argsort(x, axis=axis).reshape(-1, in_shape[axis])
    rank = np.empty_like(tmp, dtype=out_dtype)
    np.put_along_axis(rank, tmp, np.arange(1, in_shape[axis] + 1), axis=axis)
    del tmp
    rank = rank.reshape(in_shape)
    return rank


def _da_rankdata(x: dask.array.Array, axis: int = -1, out_dtype="int32"):
    """Rank data using dask.

    Avoids using argsort 2 times.
    Wraps around _np_rankdata using dask.array.map_blocks
    """
    rank_dask = dask.array.map_blocks(
        _np_rankdata,
        x,
        dtype=out_dtype,
        # _np_rankdata kwargs
        axis=axis,
        out_dtype=out_dtype,
    )
    return rank_dask


def rankdata(x: np.ndarray, axis: int = -1, out_dtype="int32"):
    """Rank data.

    Avoids using argsort 2 times.
    """
    if isinstance(x, np.ndarray):
        rank = _np_rankdata(x, axis=axis)
    elif isinstance(x, dask.array.Array):
        rank = _da_rankdata(x, axis=axis)
    elif isinstance(x, xr.DataArray):
        rank = x.copy()
        rank.data = rankdata(x.data, axis=axis)
    else:
        raise NotImplementedError()
    return rank


def _pearson_r(x, y, mean_x, mean_y, std_x, std_y):
    """
    Compute the Pearson Correlation Coefficient between pairwise columns.

    To be used on x and y 2D arrays with samples across the columns,
    and variables across the rows.

    Parameters
    ----------
    x : np.ndarray
        2D array with shape (aux, sample)
    y : np.ndarray
        2D array with shape (aux, sample)
    mean_x : np.ndarray
        Mean 2D array with shape (aux,)
    mean_y : np.ndarray
        Mean 2D array with shape (aux,)
    std_x : np.ndarray
        Standard deviation 2D array with shape (aux,)
    std_y : np.ndarray
        Standard deviation 2D array with shape (aux,)

    Returns
    -------
    corr : np.ndarray
        Correlation coefficient with shape (aux,)

    """
    mean_x = np.expand_dims(mean_x, axis=1)
    mean_y = np.expand_dims(mean_y, axis=1)

    # Compute the covariance between x and y
    cov = np.nanmean((x - mean_x) * (y - mean_y), axis=1)

    # Compute the correlation coefficient
    corr = cov / (std_x * std_y)

    # Clip to avoid numerical artefacts
    corr = np.clip(corr, -1.0, 1.0).astype("float64")

    return corr


def pearson_r(x, y):
    """
    Compute the Pearson Correlation Coefficient between pairwise columns.

    To be used on x and y 2D arrays with samples across the columns,
    and variables across the rows.

    Parameters
    ----------
    x : np.ndarray
        2D array with shape (aux, sample)
    y : np.ndarray
        2D array with shape (aux, sample)

    Returns
    -------
    corr : np.ndarray
        Correlation coefficient with shape (aux,)

    """
    corr = _pearson_r(
        x=x,
        y=y,
        mean_x=np.nanmean(x, axis=1),
        mean_y=np.nanmean(y, axis=1),
        std_x=np.nanstd(x, axis=1),
        std_y=np.nanstd(y, axis=1),
    )
    return corr


def spearman_r(x, y):
    """
    Compute the Spearman Correlation Coefficient between pairwise columns.

    To be used on x and y 2D arrays with samples across the columns,
    and variables across the rows.

    Parameters
    ----------
    x : np.ndarray
        2D array with shape (aux, sample)
    y : np.ndarray
        2D array with shape (aux, sample)

    Returns
    -------
    corr : np.ndarray
        Correlation coefficient with shape (aux,)

    """
    # TODO: implement np.nan_rankdata (now is wrong)
    corr = pearson_r(x=rankdata(x, axis=1), y=rankdata(y, axis=1))
    return corr

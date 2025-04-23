#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:47:15 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr
from xverif import EPS


##----------------------------------------------------------------------------.
# Covariance/Correlation functions for xarray
def _inner(x, y):
    result = np.matmul(x[..., np.newaxis, :], y[..., :, np.newaxis])
    return result[..., 0, 0]


def _xr_inner_product(x, y, dim, dask="parallelized"):
    if dim is not None:
        if isinstance(dim, str):
            dim = [dim]
        if isinstance(dim, tuple):
            dim = list()
        if len(dim) == 2:
            # TODO reshape data to sample_dim x 'time'
            raise NotImplementedError
        input_core_dims = [dim, dim]  # [[x_dim, y_dim]
    else:
        raise ValueError("Requiring a dimension...")
    return xr.apply_ufunc(
        _inner,
        x,
        y,
        input_core_dims=input_core_dims,
        dask="parallelized",
        output_dtypes=[float],
    )


def _xr_covariance(x, y, sample_dims=None, dask="parallelized"):
    x_mean = x.mean(sample_dims)
    y_mean = y.mean(sample_dims)
    N = x.count(sample_dims)
    return _xr_inner_product(x - x_mean, y - y_mean, dim=sample_dims, dask=dask) / N


def _xr_pearson_correlation(x, y, sample_dims=None, dask="parallelized"):
    x_std = x.std(sample_dims) + EPS
    y_std = y.std(sample_dims) + EPS
    return _xr_covariance(x, y, aggregating_sample_dims=sample_dims, dask=dask) / (
        x_std * y_std
    )


# import bottleneck
# def _xr_rank(x, dim, dask="parallelized"):
#     return xr.apply_ufunc(bottleneck.rankdata, x,
#                           input_core_dims=[[dim]],
#                           dask="parallelized")


# def _xr_spearman_correlation(x, y, sample_dims=None):
#     x_rank= x.rank(dim=sample_dims)
#     y_rank = y.rank(dim=sample_dims)
#     return _xr_pearson_correlation(x_rank,y_rank, aggregating_sample_dims=sample_dims)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:53:04 2023.

@author: ghiggi
"""
import numpy as np

#### TODO: Add option to deal with nan?


def np_rankdata(x: np.array, out_dtype="int32"):
    """Rank data using numpy.

    Avoids using argsort 2 times.
    """
    N = len(x)
    tmp = np.argsort(x)
    rank = np.empty_like(tmp, dtype=out_dtype)
    np.put_along_axis(rank, tmp, np.arange(1, N + 1), axis=0)
    del tmp
    return rank


def _pearson_r(x, y, mean_x, mean_y, std_x, std_y):
    """
    Compute the Pearson Correlation Coefficient between two 1D arrays.

    Parameters
    ----------
    x : np.array
        1D array
    y : np.darray
        1D array
    mean_x : float
        Mean value of x
    mean_y : float
        Mean value of y
    std_x : float
        Standard deviation of x
    std_y : float
        Standard deviation of y

    Returns
    -------
    corr : float
        Correlation coefficient

    """
    # Compute the covariance between x and y
    cov = np.mean((x - mean_x) * (y - mean_y))

    # Compute the correlation coefficient
    corr = cov / (std_x * std_y)

    # Clip to avoid numerical artefacts
    corr = np.clip(corr, -1.0, 1.0).astype("float64")
    return corr


def pearson_r(x, y):
    """
    Compute the Pearson Correlation Coefficient between two 1D arrays.

    Parameters
    ----------
    x : np.array
        1D array
    y : np.darray
        1D array

    Returns
    -------
    corr : float
        Pearson correlation coefficient

    """
    corr = _pearson_r(
        x=x,
        y=y,
        mean_x=np.mean(x),
        mean_y=np.mean(y),
        std_x=np.std(x),
        std_y=np.std(y),
    )
    return corr


def spearman_r(x, y):
    """
    Compute the Spearman Correlation Coefficient between two 1D arrays.

    Parameters
    ----------
    x : np.array
        1D array
    y : np.darray
        1D array

    Returns
    -------
    corr : float
        Spearmann correlation coefficient

    """
    corr = pearson_r(x=np_rankdata(x), y=np_rankdata(y))
    return corr

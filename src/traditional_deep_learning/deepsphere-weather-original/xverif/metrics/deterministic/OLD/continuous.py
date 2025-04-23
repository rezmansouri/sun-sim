#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:38:26 2023.

@author: ghiggi
"""
import numpy as np
import scipy.stats
import xarray as xr
from dask.diagnostics import ProgressBar
from xverif import EPS
from xverif.dropping import DropData
from xverif.utils.timing import print_elapsed_time


def _get_metrics(pred, obs, drop_options=None):
    """Deterministic metrics for continuous predictions forecasts.

    This function expects pred and obs to be 1D vector of same size
    """
    # Preprocess data
    pred = pred.flatten()
    obs = obs.flatten()
    pred, obs = DropData(pred, obs, drop_options=drop_options).apply()

    # If not non-NaN data, return a vector of nan data
    if len(pred) == 0:
        return np.ones(27) * np.nan
    ##------------------------------------------------------------------------.
    # - Error
    error = pred - obs
    error_squared = error**2
    error_perc = error / (obs + EPS)
    ##------------------------------------------------------------------------.
    # - Mean
    pred_mean = pred.mean()
    obs_mean = obs.mean()
    error_mean = error.mean()
    ##------------------------------------------------------------------------.
    # - Standard deviation
    pred_std = pred.std()
    obs_std = obs.std()
    error_std = error.std()
    ##------------------------------------------------------------------------.
    # - Coefficient of variability
    pred_CoV = pred_std / (pred_mean + EPS)
    obs_CoV = obs_std / (obs_mean + EPS)
    error_CoV = error_std / (error_mean + EPS)
    ##------------------------------------------------------------------------.
    # - Magnitude metrics
    BIAS = error_mean
    MAE = np.abs(error).mean()
    MSE = error_squared.mean()
    RMSE = np.sqrt(MSE)

    percBIAS = error_perc.mean() * 100
    percMAE = np.abs(error_perc).mean() * 100

    relBIAS = BIAS / (obs_mean + EPS)
    relMAE = MAE / (obs_mean + EPS)
    relMSE = MSE / (obs_mean + EPS)
    relRMSE = RMSE / (obs_mean + EPS)
    ##------------------------------------------------------------------------.
    # - Average metrics
    rMean = pred_mean / (obs_mean + EPS)
    diffMean = pred_mean - obs_mean
    ##------------------------------------------------------------------------.
    # - Variability metrics
    rSD = pred_std / (obs_std + EPS)
    diffSD = pred_std - obs_std
    rCoV = pred_CoV / obs_CoV
    diffCoV = pred_CoV - obs_CoV
    # - Correlation metrics
    pearson_R, pearson_R_pvalue = scipy.stats.pearsonr(pred, obs)
    pearson_R2 = pearson_R**2

    spearman_R, spearman_R_pvalue = scipy.stats.spearmanr(pred, obs)
    spearman_R2 = spearman_R**2

    ##------------------------------------------------------------------------.
    # - Overall skill metrics
    LTM_forecast_error = ((obs_mean - obs) ** 2).sum()  # Long-term mean as prediction
    NSE = 1 - (error_squared.sum() / (LTM_forecast_error + EPS))
    KGE = 1 - (np.sqrt((pearson_R - 1) ** 2 + (rSD - 1) ** 2 + (rMean - 1) ** 2))

    ##------------------------------------------------------------------------.
    skills = np.array(
        [
            pred_CoV,
            obs_CoV,
            error_CoV,
            # Magnitude
            BIAS,
            MAE,
            MSE,
            RMSE,
            percBIAS,
            percMAE,
            relBIAS,
            relMAE,
            relMSE,
            relRMSE,
            # Average
            rMean,
            diffMean,
            # Variability
            rSD,
            diffSD,
            rCoV,
            diffCoV,
            # Correlation
            pearson_R,
            pearson_R_pvalue,
            pearson_R2,
            spearman_R,
            spearman_R_pvalue,
            spearman_R2,
            # Overall skill
            NSE,
            KGE,
        ]
    )
    return skills


def get_metrics_info():
    """Get metrics information."""
    func = _get_metrics
    skill_names = [
        "pred_CoV",
        "obs_CoV",
        "error_CoV",
        # Magnitude
        "BIAS",
        "MAE",
        "MSE",
        "RMSE",
        "percBIAS",
        "percMAE",
        "relBIAS",
        "relMAE",
        "relMSE",
        "relRMSE",
        # Average
        "rMean",
        "diffMean",
        # Variability
        "rSD",
        "diffSD",
        "rCoV",
        "diffCoV",
        # Correlation
        "pearson_R",
        "pearson_R_pvalue",
        "pearson_R2",
        "spearman_R",
        "spearman_R_pvalue",
        "spearman_R2",
        # Overall skill
        "NSE",
        "KGE",
    ]
    return func, skill_names


@print_elapsed_time(task="deterministic continuous")
def _xr_apply_routine(
    pred,
    obs,
    sample_dims,
    **kwargs,
):
    """Compute deterministic continuous metrics."""
    # Retrieve function and skill names
    func, skill_names = get_metrics_info()

    # Check kwargs
    # TODO

    # Define gufunc kwargs
    input_core_dims = [sample_dims, sample_dims]
    dask_gufunc_kwargs = {
        "output_sizes": {
            "skill": len(skill_names),
        }
    }

    # Apply ufunc
    ds_skill = xr.apply_ufunc(
        func,
        pred,
        obs,
        kwargs=kwargs,
        input_core_dims=input_core_dims,
        output_core_dims=[["skill"]],  # returned data has one dimension
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs=dask_gufunc_kwargs,
        output_dtypes=["float64"],
    )

    # Compute the skills
    with ProgressBar():
        ds_skill = ds_skill.compute()

    # Add skill coordinates
    ds_skill = ds_skill.assign_coords({"skill": skill_names})

    # Return the skill Dataset
    return ds_skill

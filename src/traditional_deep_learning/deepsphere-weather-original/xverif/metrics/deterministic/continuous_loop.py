#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:38:26 2023.

@author: ghiggi
"""
import numpy as np

# import scipy.stats
import xarray as xr
from dask.diagnostics import ProgressBar
from xverif import EPS
from xverif.dropping import DropData
from xverif.metrics.deterministic.correlations import (
    _pearson_r,
    spearman_r,
)
from xverif.utils.timing import print_elapsed_time


def _mIOA(error, pred, obs_anomaly, obs_Mean, j=1):
    """Return the modified Index of Agreement.

    See Wilmott et al., 1981.
    """
    # TOREAD
    # https://search.r-project.org/CRAN/refmans/hydroGOF/html/md.html
    # https://www.nature.com/articles/srep19401 --> IoA_Gallo
    pred_anomaly = pred - obs_Mean
    denominator = np.sum(((np.abs(obs_anomaly) + np.abs(pred_anomaly)) ** j))
    mIoA = 1 - np.sum(error**j) / (denominator + EPS)
    return mIoA


def _get_metrics(pred, obs, drop_options=None):
    """Deterministic metrics for continuous predictions forecasts.

    This function expects pred and obs to be 1D vector of same size.
    """
    # Preprocess data
    pred = pred.flatten()
    obs = obs.flatten()
    pred, obs = DropData(pred, obs, drop_options=drop_options).apply()

    # If not non-NaN data, return a vector of nan data
    if len(pred) == 0:
        return np.ones(67) * np.nan

    ##------------------------------------------------------------------------.
    # Number of valid samples
    N = np.logical_and(np.isfinite(pred), np.isfinite(obs)).sum()

    ##------------------------------------------------------------------------.
    # - Error
    error = pred - obs
    error_squared = error**2
    error_abs = np.abs(error)
    error_perc = error / (obs + EPS)
    error_abs_perc = error_abs / np.maximum(np.abs(obs), EPS)

    ##------------------------------------------------------------------------.
    # Sum of errors
    SE = np.sum(error)
    SAE = np.sum(error_abs)
    SSE = np.sum(error_squared)

    ##------------------------------------------------------------------------.
    # - Mean
    pred_Mean = np.mean(pred)
    obs_Mean = np.mean(obs)
    error_Mean = np.mean(error)

    obs_anomaly = obs - obs_Mean

    ##------------------------------------------------------------------------.
    # - Standard deviation
    pred_SD = np.std(pred)
    obs_SD = np.std(obs)
    error_SD = np.std(error)

    ##------------------------------------------------------------------------.
    # - Coefficient of variability
    pred_CoV = pred_SD / (pred_Mean + EPS)
    obs_CoV = obs_SD / (obs_Mean + EPS)
    error_CoV = error_SD / (error_Mean + EPS)

    ##------------------------------------------------------------------------.
    # - Magnitude metrics
    BIAS = error_Mean
    MAE = np.mean(error_abs)
    MSE = np.mean(error_squared)
    RMSE = np.sqrt(MSE)

    MaxAbsError = np.max(error_abs)

    percBIAS = np.mean(error_perc) * 100
    percMAE = np.mean(np.abs(error_perc)) * 100  # MAPE
    MAPE = np.mean(error_abs_perc) * 100

    relBIAS = BIAS / (obs_Mean + EPS)
    relMAE = MAE / (obs_Mean + EPS)
    relMSE = MSE / (obs_Mean + EPS)
    relRMSE = RMSE / (obs_Mean + EPS)

    ##------------------------------------------------------------------------.
    # - Average metrics
    rMean = pred_Mean / (obs_Mean + EPS)
    diffMean = pred_Mean - obs_Mean

    ##------------------------------------------------------------------------.
    # - Variability metrics
    rSD = pred_SD / (obs_SD + EPS)
    diffSD = pred_SD - obs_SD
    rCoV = pred_CoV / obs_CoV
    diffCoV = pred_CoV - obs_CoV

    # Explained Variance Score
    # - Variance accounted for (VAF)
    EVS = 1 - error_SD**2 / obs_SD**2

    ##------------------------------------------------------------------------.
    # - Correlation metrics
    # pearson_R, pearson_R_pvalue = scipy.stats.pearsonr(pred, obs)
    # spearman_R, spearman_R_pvalue = scipy.stats.spearmanr(pred, obs)
    pearson_R = _pearson_r(
        x=pred, y=obs, mean_x=pred_Mean, mean_y=obs_Mean, std_x=pred_SD, std_y=obs_SD
    )
    pearson_R2 = pearson_R**2
    spearman_R = spearman_r(x=pred, y=obs)
    spearman_R2 = spearman_R**2

    ##------------------------------------------------------------------------.
    # - Overall skill metrics
    # - Nash Sutcliffe efficiency / Reduction of variance
    # - --> 1 - MSE/Var(obs)
    denominator = np.sum(obs_anomaly**2)
    NSE = 1 - (SSE / (denominator + EPS))

    # - Klinga Gupta Efficiency Score
    KGE = 1 - (np.sqrt((pearson_R - 1) ** 2 + (rSD - 1) ** 2 + (rMean - 1) ** 2))

    # - Modified Index of Agreement (Wilmott et al., 1981)
    mIoA1 = _mIOA(error, pred, obs_anomaly, obs_Mean, j=1)
    mIoA2 = _mIOA(error, pred, obs_anomaly, obs_Mean, j=2)
    mIoA3 = _mIOA(error, pred, obs_anomaly, obs_Mean, j=3)

    ##------------------------------------------------------------------------.
    # - Median
    obs_q25, obs_Median, obs_q75 = np.quantile(obs, q=[0.25, 0.50, 0.75])
    pred_q25, pred_Median, pred_q75 = np.quantile(pred, q=[0.25, 0.50, 0.75])
    error_q25, error_Median, error_q75 = np.quantile(error, q=[0.25, 0.50, 0.75])

    # pred_Median = np.median(pred)
    # obs_Median = np.median(obs)
    # error_Median = np.median(error)

    ##------------------------------------------------------------------------.
    # TODO: Scatter
    # - Scatter
    # --> Half the distance between the 16% and 84% percentiles with error dB(pred/obs)
    # error_db = 10 * np.log10(pred / (obs + EPS))
    # q16, q25, q50, q75, q84 = np.quantile(error_db, q=[0.16, 0.25, 0.5, 0.75, 0.84])
    # SCATTER = (q84 - q16) / 2.0

    ##------------------------------------------------------------------------.
    # - Mean Absolute Deviation
    pred_MAD = np.median(np.abs(pred - pred_Median))
    obs_MAD = np.median(np.abs(obs - obs_Median))
    error_MAD = np.median(np.abs(error - error_Median))

    ##------------------------------------------------------------------------.
    # - Interquartile range
    pred_IQR = pred_q75 - pred_q25
    obs_IQR = obs_q75 - obs_q25
    error_IQR = error_q75 - error_q25

    ##------------------------------------------------------------------------.
    # - Robust Coefficient of variability
    rob_pred_CoV = pred_MAD / (pred_Median + EPS)
    rob_obs_CoV = obs_MAD / (obs_Median + EPS)
    rob_error_CoV = error_MAD / (error_Median + EPS)

    ##------------------------------------------------------------------------.
    # - Robust Magnitude metrics
    rob_BIAS = error_Median
    rob_MAE = np.median(error_abs)
    rob_MSE = np.median(error_squared)
    rob_RMSE = np.sqrt(rob_MSE)

    rob_percBIAS = np.median(error_perc) * 100
    rob_percMAE = np.median(np.abs(error_perc)) * 100

    rob_relBIAS = rob_BIAS / (obs_Median + EPS)
    rob_relMAE = rob_MAE / (obs_Median + EPS)
    rob_relMSE = rob_MSE / (obs_Median + EPS)
    rob_relRMSE = rob_RMSE / (obs_Median + EPS)

    ##------------------------------------------------------------------------.
    # - Average metrics
    rMedian = pred_Median / (obs_Median + EPS)
    diffMedian = pred_Median - obs_Median

    ##------------------------------------------------------------------------.
    # - Variability metrics
    rMAD = pred_MAD / (obs_MAD + EPS)
    diffMAD = pred_MAD - obs_MAD
    rob_rCoV = rob_pred_CoV / rob_obs_CoV
    rob_diffCoV = rob_pred_CoV - rob_obs_CoV

    ##------------------------------------------------------------------------.
    dictionary = {
        "N": N,
        # Sum of errors
        "SE": SE,
        "SAE": SAE,
        "SSE": SSE,
        # Mean
        "obs_Mean": obs_Mean,
        "pred_Mean": pred_Mean,
        "rMean": rMean,
        "diffMean": diffMean,
        # Median
        "obs_Median": obs_Median,
        "pred_Median": pred_Median,
        "rMedian": rMedian,
        "diffMedian": diffMedian,
        # Variability
        "obs_SD": obs_SD,
        "pred_SD": pred_SD,
        "error_SD": error_SD,
        "diffSD": diffSD,
        "rSD": rSD,
        "obs_CoV": obs_CoV,
        "pred_CoV": pred_CoV,
        "error_CoV": error_CoV,
        "rCoV": rCoV,
        "diffCoV": diffCoV,
        "EVS": EVS,
        "obs_MAD": obs_MAD,
        "pred_MAD": pred_MAD,
        "error_MAD": error_MAD,
        "obs_IQR": obs_IQR,
        "pred_IQR": pred_IQR,
        "error_IQR": error_IQR,
        "diffMAD": diffMAD,
        "rMAD": rMAD,
        "rob_obs_CoV": rob_obs_CoV,
        "rob_pred_CoV": rob_pred_CoV,
        "rob_error_CoV": rob_error_CoV,
        "rob_diffCoV": rob_diffCoV,
        "rob_rCoV": rob_rCoV,
        # BIAS
        "BIAS": BIAS,
        "percBIAS": percBIAS,
        "relBIAS": relBIAS,
        "rob_BIAS": rob_BIAS,
        "rob_percBIAS": rob_percBIAS,
        "rob_relBIAS": rob_relBIAS,
        # Magnitude error
        "MaxAbsError": MaxAbsError,
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE,
        "rob_MAE": rob_MAE,
        "rob_MSE": rob_MSE,
        "rob_RMSE": rob_RMSE,
        "percMAE": percMAE,
        "MAPE": MAPE,
        "rob_percMAE": rob_percMAE,
        "relMAE": relMAE,
        "relMSE": relMSE,
        "relRMSE": relRMSE,
        "rob_relMAE": rob_relMAE,
        "rob_relMSE": rob_relMSE,
        "rob_relRMSE": rob_relRMSE,
        # Correlations
        "pearson_R": pearson_R,
        "pearson_R2": pearson_R2,
        "spearman_R": spearman_R,
        "spearman_R2": spearman_R2,
        # Overall
        "NSE": NSE,
        "KGE": KGE,
        "mIoA1": mIoA1,
        "mIoA2": mIoA2,
        "mIoA3": mIoA3,
    }
    skills = np.array(list(dictionary.values()))
    # metrics = list(dictionary)
    return skills


def get_metrics_info():
    """Get metrics information."""
    func = _get_metrics
    skill_names = [
        "N",
        "SE",
        "SAE",
        "SSE",
        "obs_Mean",
        "pred_Mean",
        "rMean",
        "diffMean",
        "obs_Median",
        "pred_Median",
        "rMedian",
        "diffMedian",
        "obs_SD",
        "pred_SD",
        "error_SD",
        "diffSD",
        "rSD",
        "obs_CoV",
        "pred_CoV",
        "error_CoV",
        "rCoV",
        "diffCoV",
        "EVS",
        "obs_MAD",
        "pred_MAD",
        "error_MAD",
        "obs_IQR",
        "pred_IQR",
        "error_IQR",
        "diffMAD",
        "rMAD",
        "rob_obs_CoV",
        "rob_pred_CoV",
        "rob_error_CoV",
        "rob_diffCoV",
        "rob_rCoV",
        "BIAS",
        "percBIAS",
        "relBIAS",
        "rob_BIAS",
        "rob_percBIAS",
        "rob_relBIAS",
        "MaxAbsError",
        "MAE",
        "MSE",
        "RMSE",
        "rob_MAE",
        "rob_MSE",
        "rob_RMSE",
        "percMAE",
        "MAPE",
        "rob_percMAE",
        "relMAE",
        "relMSE",
        "relRMSE",
        "rob_relMAE",
        "rob_relMSE",
        "rob_relRMSE",
        "pearson_R",
        "pearson_R2",
        "spearman_R",
        "spearman_R2",
        "NSE",
        "KGE",
        "mIoA1",
        "mIoA2",
        "mIoA3",
    ]
    return func, skill_names


@print_elapsed_time(task="deterministic continuous")
def _xr_apply_routine(
    pred,
    obs,
    sample_dims,
    metrics=None,
    compute=True,
    drop_options=None,
):
    """Compute deterministic continuous metrics.

    With this implementation all metrics are computed and then subsetted.
    """
    # Retrieve function and skill names
    func, skill_names = get_metrics_info()

    # Define kwargs
    kwargs = {}
    kwargs["drop_options"] = drop_options

    # Define gufunc kwargs
    input_core_dims = [sample_dims, sample_dims]
    dask_gufunc_kwargs = {
        "output_sizes": {
            "skill": len(skill_names),
        }
    }

    # Apply ufunc
    da_skill = xr.apply_ufunc(
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
    if compute:
        with ProgressBar():
            da_skill = da_skill.compute()

    # Add skill coordinates
    da_skill = da_skill.assign_coords({"skill": skill_names})

    # Subset skill coordinates
    # TODO

    # Convert to skill Dataset
    ds_skill = da_skill.to_dataset(dim="skill")

    # Return the skill Dataset
    return ds_skill

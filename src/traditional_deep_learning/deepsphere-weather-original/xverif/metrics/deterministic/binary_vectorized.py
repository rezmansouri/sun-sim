#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:38:26 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from xverif import EPS
from xverif.utils.timing import print_elapsed_time
from xverif.utils.warnings import suppress_warnings


def get_aux_dims(pred, sample_dims):
    """Infer aux_dims from prediction dataset.

    Keep order of the dataset dimensions!
    """
    dims = list(pred.dims)
    aux_dims = [dim for dim in dims if dim not in sample_dims]
    # aux_dims = set(dims).difference(sample_dims)
    return aux_dims


def get_stacking_dict(pred, sample_dims):
    """Get stacking dimension dictionary."""
    aux_dims = get_aux_dims(pred, sample_dims)
    stacking_dict = {
        "stacked_dim": aux_dims,
        "sample_dim": sample_dims,
    }
    return stacking_dict


def get_available_metrics():
    """Return available metrics."""
    arr = np.zeros((1, 3))
    dict_metrics = _get_metrics(arr, arr)
    return list(dict_metrics)


def check_metrics(metrics):
    """Checks metrics validity.

    If None, returns all available metrics.
    """
    available_metrics = get_available_metrics()
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(metrics, type(None)):
        metrics = available_metrics
    if isinstance(metrics, (tuple, list)):
        metrics = np.array(metrics, dtype="str")

    idx_invalid = np.where(np.isin(metrics, available_metrics, invert=True))[0]
    if len(idx_invalid) > 0:
        invalid_metrics = metrics[idx_invalid]
        raise ValueError(f"These metrics are invalid: {invalid_metrics}")
    return metrics


@suppress_warnings
def _get_metrics(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Deterministic metrics for binary forecasts.

    Parameters
    ----------
    pred : np.ndarray
        2D prediction array of shape (aux, sample)
        The columns corresponds to the sample predictions to be used
        to compute the metrics (for each row).
    obs : np.ndarray
        2D observation array with same shape as pred: (aux, sample).
        The columns corresponds to the sample observations to be used
        to compute the metrics (for each row).

    Returns
    -------
    skills : dict
        A dictionary of format <metric>: <value>.
    """
    # Define axis
    axis = 1

    # Calculate number hits, misses, false alarms, correct rejects
    H = np.nansum(np.logical_and(pred == 1, obs == 1), axis=axis, dtype="float64")
    M = np.nansum(np.logical_and(pred == 0, obs == 1), axis=axis, dtype="float64")
    F = np.nansum(np.logical_and(pred == 1, obs == 0), axis=axis, dtype="float64")
    R = np.nansum(np.logical_and(pred == 0, obs == 0), axis=axis, dtype="float64")

    # Number of samples
    N = H + F + M + R

    # Correct predictions
    N_correct = H + R

    # Wrong predictions
    N_wrong = F + M

    # True/False Positive/Negative
    TP = H
    FN = M
    FP = F
    TN = R

    # Probability of detection
    # - Detection Rate
    # - Sensitivity
    # - Recall (score)
    # - True Positive Rate
    # - Hit Rate
    POD = H / (H + M + EPS)

    # Probability of Rejection
    # - Specificity
    POR = R / (F + R + EPS)

    # Precision
    # - Success Ratio
    precision = H / (H + F + EPS)

    # Correct-Rejection Ratio
    # - Negative Predictive Value
    CRR = R / (M + R + EPS)

    # Probability of False Rejection
    # - Miss Rate
    PFR = M / (H + M + EPS)

    # Miss Ratio
    MR = M / (M + R + EPS)

    # False Alarm Rate
    # - Prob of false detection (POFD)
    # - Fall-out
    # - False positive rate
    FA = F / (F + R + EPS)

    # False Alarm Ratio
    # - False Discovery Rate
    FAR = F / (H + F + EPS)

    # Success Ratio (SR)
    # - Hit Ratio (HR)
    SR = 1 - FAR

    # Youden J statistics
    # - Informedness
    Informedness = POD + POR - 1

    # Markdness
    Markedness = SR + CRR - 1

    # Hanssen-Kuipers Discriminant
    # - Peirce Skill Score (PSS)
    # - True Skill Statistics (TSS)
    HK = POD - FA

    # Standard deviations
    FA_std = np.sqrt(FA * (1 - FA) / (F + R + EPS))
    POD_std = np.sqrt(POD * (1 - POD) / (H + M + EPS))
    FAR_std = np.sqrt(
        (FAR**4)
        * ((1 - POD) / (H + EPS) + (1 - FA) / (F + EPS))
        * (H**2)
        / (F**2 + EPS)
    )
    SR_std = FAR_std

    # Critical Success Index
    # - Threat Score
    CSI = H / (H + N_wrong + EPS)
    CSI_std = np.sqrt(
        (CSI**2) * ((1 - POD) / (H + EPS) + F * (1 - FA) / ((H + N_wrong) ** 2 + EPS))
    )

    # Frequency Bias
    FB = (H + F) / (H + M + EPS)

    # Actual positives
    s = (H + M) / N

    # Hamming Loss
    # - Zero-One Loss
    # - Overall error rate
    # - 1−ACC
    HL = N_wrong / N

    # Accuracy
    # - Fraction correct
    # - Accuracy score
    # - Overall accuracy (OA)
    # - Percent correct (PC)
    # - Exact match Ratio (EMR) (?)
    # - Hamming score
    ACC = (N_correct) / N
    ACC_std = np.sqrt(s * POD * (1 - POD) / N + (1 - s) * FA * (1 - FA) / N)

    # Heidke Skill Score (-1 < HSS < 1) < 0 implies no skill
    # - Cohen’s Kappa
    HSS = 2 * (H * R - F * M) / ((H + M) * (M + R) + (H + F) * (F + R) + EPS)
    HSS_std = np.sqrt(
        (FA_std**2) * (HSS**2) * (1 / (POD - FA + EPS) + (1 - s) * (1 - 2 * s)) ** 2
        + (POD_std**2) * (HSS**2) * (1 / (POD - FA + EPS) - s * (1 - 2 * s)) ** 2
    )

    # Equitable Threat Score (ETS)
    # -  Gilbert Skill Score (GSS)
    # --> TODO: check equality
    # ETS = (POD - FA) / ((1 - s * POD) / (1 - s) + FA * (1 - s) / s)
    HITSrandom = 1 * (H + M) * (H + F) / N
    ETS = (H - HITSrandom) / (H + N_wrong - HITSrandom + EPS)
    ETS_std = np.sqrt(4 * (HSS_std**2) / ((2 - HSS + EPS) ** 4))

    # F1 score
    # - Dice Coefficient
    # - The harmonic mean of precision and sensitivity (pysteps)
    F1 = 2 * H / (2 * H + N_wrong + EPS)
    # F1 = 2 * (precision * POD) / (precision + POD)

    # F2 score
    # - 2x emphasis on recall.
    F2 = 5 * (precision * POD) / (4 * precision + POD)

    # Geometric Mean Score (GMS)
    GMS = np.sqrt(POD * POR)

    # Jaccard Index
    # - Tanimoto Coefficient
    # - Intersection over Union (IoU)
    # J = H / (H + N_wrong + EPS)
    J = F1 / (2 - F1)

    # Matthews Correlation Coefficient
    # # TODO CHECK
    # - if denominator 0, result should be? 0?
    MCC = (H * R - F * M) / np.sqrt((H + F) * (H + M) * (R + F) * (R + M))
    MCC = (H * R) / np.sqrt((H + F) * (H + M) * (R + F) * (R + M))

    # Lift score
    LS = (TP / (TP + FP)) / ((TP + FN) / N)

    # Odds Ratio and Log Odds Ratio
    OR = H * R / (F * M + EPS)
    LOR = np.log(OR)  # LOR = np.log(H) + np.log(R) - np.log(F) - np.log(M)

    ## Odds Ratio Skill Score (ORSS)
    # - Yules's Q
    YulesQ = (OR - 1) / (OR + 1)  # TODO check
    ORSS = (H * R - F * M) / (H * R + F * M + EPS)
    n_h = 1 / (1 / H + 1 / F + 1 / M + 1 / R + EPS)  # Error if H, F, M or R = 0
    ORSS_std = np.sqrt(1 / n_h * 4 * OR**2 / ((OR + 1) ** 4))

    # (Symmetric) Extreme Dependency Score (EDS)
    p = (H + M) / N
    EDS = 2 * np.log((H + M) / N) / np.log(H / N) - 1  # Error when H==N, H+M=0
    SEDS = (np.log((H + F) / N) + np.log((H + M) / N)) / np.log(
        H / N
    ) - 1  # Error when H==N, H+M=0, H+F=0
    EDS_std = (
        2
        * np.abs(np.log(p))
        / (POD * (np.log(p) + np.log(POD)) ** 2)
        * np.sqrt(POD * (1 - POD) / (p * N))
    )
    SEDS_std = np.sqrt(POD * (1 - POD) / (N * p)) * (
        -np.log(FB * p**2) / (POD * np.log(POD * p) ** 2)
    )

    # (Symmetric) Extremal Dependence Index (SEDI)
    # - Error when FA or POD = 0
    EDI = (np.log(FA) - np.log(POD)) / (np.log(FA) + np.log(POD) + EPS)
    SEDI = (np.log(FA) - np.log(POD) + np.log(1 - POD) - np.log(1 - FA)) / (
        np.log(FA) + np.log(POD) + np.log(1 - POD) + np.log(1 - FA)
    )

    EDI_std = (
        2
        * np.abs(np.log(FA) + POD / (1 - POD) * np.log(POD))
        / (POD * (np.log(FA) + np.log(POD)) ** 2)
        * np.sqrt(POD * (1 - POD) / (p * N))
    )
    SEDI_std = (
        2
        * np.abs(
            ((1 - POD) * (1 - FA) + POD * FA)
            / ((1 - POD) * (1 - FA))
            * np.log(FA * (1 - POD))
            + 2 * POD / (1 - POD) * np.log(POD * (1 - FA))
        )
        / (POD * (np.log(FA * (1 - POD)) + np.log(POD * (1 - FA))) ** 2)
        * np.sqrt(POD * (1 - POD) / (p * N))
    )

    # Define metrics
    dictionary = {
        "N": N,
        "N_correct": N_correct,
        "N_wrong": N_wrong,
        "H": H,
        "F": F,
        "M": M,
        "R": R,
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "TN": TN,
        "POD": POD,
        "POD_std": POD_std,
        "FAR": FAR,
        "FAR_std": FAR_std,
        "FA": FA,
        "FA_std": FA_std,
        "SR": SR,
        "SR_std": SR_std,
        "POR": POR,
        "PFR": PFR,
        "MR": MR,
        "CRR": CRR,
        "GMS": GMS,
        "Informedness": Informedness,
        "Markedness": Markedness,
        "FB": FB,
        "HK": HK,
        "HL": HL,
        "ACC": ACC,
        "ACC_std": ACC_std,
        "CSI": CSI,
        "CSI_std": CSI_std,
        "HSS": HSS,
        "HSS_std": HSS_std,
        "ETS": ETS,
        "ETS_std": ETS_std,
        "OR": OR,
        "LOR": LOR,
        "ORSS": ORSS,
        "ORSS_std": ORSS_std,
        "MCC": MCC,
        "F1": F1,
        "F2": F2,
        "J": J,
        "LS": LS,
        "YulesQ": YulesQ,
        "EDS": EDS,
        "EDS_std": EDS_std,
        "SEDS": SEDS,
        "SEDS_std": SEDS_std,
        "EDI": EDI,
        "EDI_std": EDI_std,
        "SEDI": SEDI,
        "SEDI_std": SEDI_std,
    }
    return dictionary


def get_metrics(
    pred: np.ndarray, obs: np.ndarray, metrics: np.ndarray, **kwargs
) -> np.ndarray:
    """
    Deterministic metrics for binary forecasts.

    Parameters
    ----------
    pred : np.ndarray
        2D prediction array of shape (aux, sample)
        The columns corresponds to the sample predictions to be used
        to compute the metrics (for each row).
    obs : TYPE
        2D observation array with same shape as pred: (aux, sample).
        The columns corresponds to the sample observations to be used
        to compute the metrics (for each row).
    metrics : np.ndarray
        List of metrics to compute.

    Returns
    -------
    skills : np.ndarray
        A 2D array of shape (aux, n_skills).
    """
    dict_metrics = _get_metrics(pred=pred, obs=obs)
    list_skills = [dict_metrics[metric] for metric in metrics]
    skills = np.stack(list_skills, axis=-1)
    return skills


@print_elapsed_time(task="deterministic binary")
def _xr_apply_routine(
    pred,
    obs,
    sample_dims,
    metrics=None,
    compute=True,
    **kwargs,
):
    """Routine to compute metrics in a vectorized way.

    The input xarray objects are first vectorized to have 2D dimensions: (aux, sample).
    Then custom preprocessing is applied to deal with NaN, non-finite values,
    samples with equals values (i.e. 0) or samples outside specific value ranges.
    The resulting array is then passed to the function computing the metrics.

    Each chunk of the xr.DataArray is processed in parallel using dask, and the
    results recombined using xr.apply_ufunc.
    """
    metrics = check_metrics(metrics)

    # Broadcast obs to pred
    # - Creates a view, not a copy !
    # obs = obs.broadcast_like(
    #     pred
    # )  # TODO: broadcast Dataset ... so to preprocessing per variable !
    # obs_broadcasted['var0'].data.flags # view (Both contiguous are False)
    # obs_broadcasted['var0'].data.base  # view (Return array and not None)

    # Ensure obs same chunks of pred on auxiliary dims
    # TODO: now makes same as pred --> TO IMPROVE
    obs = obs.chunk(pred.chunks)

    # Stack pred and obs to have 2D dimensions (aux, sample)
    # - This operation doubles the memory (!!!)
    stacking_dict = get_stacking_dict(pred, sample_dims=sample_dims)
    pred = pred.stack(stacking_dict)
    obs = obs.stack(stacking_dict)

    # Preprocess
    # - Based on kwargs

    # Define gufunc kwargs
    input_core_dims = [["sample_dim"], ["sample_dim"]]
    dask_gufunc_kwargs = {
        "output_sizes": {
            "skill": len(metrics),
        }
    }

    # Apply ufunc
    da_skill = xr.apply_ufunc(
        get_metrics,
        pred,
        obs,
        kwargs={"metrics": metrics},
        input_core_dims=input_core_dims,
        output_core_dims=[["skill"]],
        vectorize=False,
        dask="allowed",
        dask_gufunc_kwargs=dask_gufunc_kwargs,
        output_dtypes=["float64"],
    )

    # Compute the skills
    if compute:
        with ProgressBar():
            da_skill = da_skill.compute()

    # Add skill coordinates
    da_skill = da_skill.assign_coords({"skill": metrics})

    # Unstack auxiliary dimensions
    da_skill = da_skill.unstack("stacked_dim")

    # Convert to skill Dataset
    ds_skill = da_skill.to_dataset(dim="skill")

    # Return the skill Dataset
    return ds_skill

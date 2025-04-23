#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:45:52 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from xverif import EPS
from xverif.dropping import DropData
from xverif.utils.timing import print_elapsed_time

# Rename
# - N_correct = H + R
# - N_wrong = F+M


def _get_metrics(pred, obs, drop_options=None):
    """Compute deterministic metrics for binary predictions.

    This function expects pred and obs to be 1D vector of same size.
    """
    # Preprocess data
    pred = pred.flatten()
    obs = obs.flatten()
    pred, obs = DropData(pred, obs, drop_options=drop_options).apply()

    # If not non-NaN data, return a vector of nan data
    if len(pred) == 0:
        return np.ones(55) * np.nan

    # Calculate number hits, misses, false alarms, correct rejects
    H = np.nansum(np.logical_and(pred == 1, obs == 1), dtype="float64")
    M = np.nansum(np.logical_and(pred == 0, obs == 1), dtype="float64")
    F = np.nansum(np.logical_and(pred == 1, obs == 0), dtype="float64")
    R = np.nansum(np.logical_and(pred == 0, obs == 0), dtype="float64")

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
    # - TODO Success ratio (SR?)
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
    # HK = POD + POR - 1
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

    # Accuracy (fraction correct)
    # - Fraction correct
    # - Accuracy score
    # - Overall accuracy (OA)
    # - Percent correct (PC)
    # - Exact match Ratio (EMR) (?)
    # - Hamming score
    ACC = N_correct / N
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

    skills = np.array(list(dictionary.values()))
    # metrics = list(dictionary)
    return skills


##----------------------------------------------------------------------------.


def get_metrics_info():
    """Get metrics information."""
    func = _get_metrics
    skill_names = [
        "N",
        "N_correct",
        "N_wrong",
        "H",
        "F",
        "M",
        "R",
        "TP",
        "FN",
        "FP",
        "TN",
        "POD",
        "POD_std",
        "FAR",
        "FAR_std",
        "FA",
        "FA_std",
        "SR",
        "SR_std",
        "POR",
        "PFR",
        "MR",
        "CRR",
        "GMS",
        "Informedness",
        "Markedness",
        "FB",
        "HK",
        "HL",
        "ACC",
        "ACC_std",
        "CSI",
        "CSI_std",
        "HSS",
        "HSS_std",
        "ETS",
        "ETS_std",
        "OR",
        "LOR",
        "ORSS",
        "ORSS_std",
        "MCC",
        "F1",
        "F2",
        "J",
        "LS",
        "YulesQ",
        "EDS",
        "EDS_std",
        "SEDS",
        "SEDS_std",
        "EDI",
        "EDI_std",
        "SEDI",
        "SEDI_std",
    ]
    return func, skill_names


@print_elapsed_time(task="deterministic binary")
def _xr_apply_routine(
    pred,
    obs,
    sample_dims,
    metrics=None,
    compute=True,
    drop_options=None,
):
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
    )  # dtype

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

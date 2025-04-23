#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:43:11 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from xskillscore import crps_ensemble


def _prob_metrics(pred, obs, dims, crps_ref, dim_member="member"):
    ##------------------------------------------------------------------------.
    # - Percentage of observations within ensemble range (WER)
    min_pred = pred.min(axis=0)
    max_pred = pred.max(axis=0)

    within_range = (obs >= min_pred) & (obs <= max_pred)
    within_range_mean = within_range.mean(dim=dims)

    ##------------------------------------------------------------------------.
    # - CRPS (Continuous Ranked Probability Score)
    dims = list(pred.dims).copy()
    dims.remove(dim_member)
    dims.remove("lead_time")
    crps = crps_ensemble(obs, pred, dim=dims)

    ##------------------------------------------------------------------------.
    # - CRPSS (Continuous Ranked Probability Skill Score)
    crpss = 1 - (crps / crps_ref)

    skills = np.array([within_range_mean, crps, crpss])

    return skills


##----------------------------------------------------------------------------.
def _xr_apply_routine(pred, obs, dims, crps_ref, dim_member="member"):
    ds_skill = xr.apply_ufunc(
        _prob_metrics,
        pred,
        obs,
        kwargs={
            "dim_member": dim_member,
            "crps_ref": crps_ref,
        },
        input_core_dims=[["member", "time"], ["time"]],
        output_core_dims=[["skill"]],  # returned data has one dimension
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {
                "skill": 3,
            }
        },
        output_dtypes=["float64"],
    )  # dtype
    # Compute the skills
    with ProgressBar():
        ds_skill = ds_skill.compute()
    # Add skill coordinates
    skill_str = ["WER", "CRPS", "CRPSS"]
    ds_skill = ds_skill.assign_coords({"skill": skill_str})
    ##------------------------------------------------------------------------.
    # Return the skill Dataset
    return ds_skill

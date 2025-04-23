#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:48:10 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from skimage.metrics import structural_similarity as ssim
from xverif.metrics.deterministic.pysteps_spatial_metrics import fss
from xverif.utils.timing import print_elapsed_time


def _spatial_metrics(pred, obs, win_size=5, thr=0.1):
    pred = pred.copy()
    obs = obs.copy()

    pred[np.isnan(pred)] = 0
    obs[np.isnan(obs)] = 0
    # - Fraction Skill Score
    fraction_ss = fss(pred, obs, thr, win_size)

    # - Structural Similarity index
    mssim = ssim(pred, obs, win_size=win_size)

    skills = np.array([mssim, fraction_ss])

    return skills


@print_elapsed_time(task="deterministic spatial")
def _xr_apply_routine(pred, obs, dim="time", thr=0.000001, win_size=5):
    input_core_dims = [[dim], [dim]] if type(dim) != list else [dim, dim]
    ds_skill = xr.apply_ufunc(
        _spatial_metrics,
        pred,
        obs,
        kwargs={"win_size": win_size, "thr": thr},
        input_core_dims=input_core_dims,
        output_core_dims=[["skill"]],  # returned data has one dimension
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {
                "skill": 2,
            }
        },
        output_dtypes=["float64"],
    )  # dtype
    # Compute the skills
    with ProgressBar():
        ds_skill = ds_skill.compute()
    # Add skill coordinates
    skill_str = ["SSIM", "FSS"]
    ds_skill = ds_skill.assign_coords({"skill": skill_str})
    ##------------------------------------------------------------------------.
    # Return the skill Dataset
    return ds_skill

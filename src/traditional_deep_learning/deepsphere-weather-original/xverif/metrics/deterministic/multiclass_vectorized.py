#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:25:08 2023.

@author: ghiggi
"""
from xverif.metrics.deterministic.multiclass_loop import get_one_vs_all_data_array


def get_class_specific_binary_metrics(pred, obs, n_categories, sample_dims):
    """Get class-specific binary metrics using one-vs-all approach."""
    from xverif.wrappers import deterministic

    pred_binary = get_one_vs_all_data_array(pred, n_categories=n_categories)
    obs_binary = get_one_vs_all_data_array(obs, n_categories=n_categories)
    ds_skills = deterministic(
        pred=pred_binary,
        obs=obs_binary,
        data_type="binary",
        sample_dims=sample_dims,
        implementation="vectorized",
        # metrics=metrics,
        # skip_options=skip_options,
    )
    return ds_skills

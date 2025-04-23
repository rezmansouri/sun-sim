#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:34:45 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from xverif.dropping import DropData
from xverif.utils.timing import print_elapsed_time

### To vectorize:
# - np.diag
# - np.bincount


def get_class_specific_binary_metrics(pred, obs, n_categories, sample_dims):
    """Get class-specific binary metrics using one-vs-all approach."""
    from xverif.wrappers import deterministic

    pred_binary = get_one_vs_all_data_array(pred, n_categories=n_categories)
    obs_binary = get_one_vs_all_data_array(obs, n_categories=n_categories)
    # Loop
    dict_skills = {}
    for i in range(n_categories):
        dict_skills[i] = deterministic(
            pred=pred_binary.isel(category=i),
            obs=obs_binary.isel(category=i),
            data_type="binary",
            sample_dims=sample_dims,
            implementation="loop",
            # metrics=metrics,
            # Preprocessing
            # skip_options=skip_options,
        )

    ds_skills = xr.concat(dict_skills.values(), dim="category")
    ds_skills["category"] = pred_binary["category"]

    return ds_skills


def get_one_vs_all_data_array(data_array, n_categories):
    """Expand a multicategory DataArray using the one-vs-all approach."""
    categories = np.arange(0, n_categories)
    new_data_array = xr.concat(
        [(data_array == category).astype(float) for category in categories],
        dim="category",
    )
    new_data_array["category"] = categories
    # Mask original nan
    isnan = np.isnan(data_array).broadcast_like(new_data_array)
    new_data_array.data[isnan.data] = np.nan
    return new_data_array


def get_multiclass_confusion_matrix(pred, obs, n_categories):
    """Compute a multiclass confusion matrix.

    Row: obs , Column: pred
    """
    # Calculate linear indices
    indices = obs * n_categories + pred

    # Compute bincount and reshape to get the confusion matrix
    mat = np.bincount(indices, minlength=n_categories**2).reshape(
        n_categories, n_categories
    )

    # TODO: ensure that row is pobs, column is pred !
    return mat


def _get_metrics(
    pred, obs, n_categories, misclassification_weights=None, drop_options=None
):
    """Compute deterministic metrics for binary predictions.

    This function expects pred and obs to be 1D vector of same size.
    """
    # Preprocess data
    pred = pred.flatten()
    obs = obs.flatten()
    pred, obs = DropData(pred, obs, drop_options=drop_options).apply()

    # If not non-NaN data, return a vector of nan data
    if len(pred) == 0:
        return np.ones(6) * np.nan  # TODO !

    pred = pred.astype("int64")
    obs = obs.astype("int64")

    # Compute the confusion matrix
    conf_matrix = get_multiclass_confusion_matrix(pred, obs, n_categories=n_categories)

    # Number of samples
    N = np.sum(conf_matrix)

    # Correct predictions
    N_correct = np.sum(np.diag(conf_matrix))

    # Wrong predictions
    N_wrong = N - N_correct

    # Hamming Loss
    # - Zero-One Loss
    # - Overall error rate
    # - 1−ACC
    error_rate = N_wrong / N

    # Accuracy (ACC)
    # - Fraction correct
    # - Accuracy score
    # - Overall accuracy (OA)
    # - Percent correct (PC)
    # - Exact match Ratio (EMR) (?)
    # - Hamming score
    accuracy = N_correct / N

    # Balanced_accuracy
    # - The macro-average of recall scores per class
    # - Each sample is weighted according to the inverse prevalence of its true class
    # - Balanced datasets, the score is equal to accuracy
    # - Worse value is 0
    per_class_correct = np.diag(conf_matrix)
    per_class_predicted = conf_matrix.sum(axis=1)
    per_class_recalls = per_class_correct / per_class_predicted
    balanced_accuracy = np.mean(per_class_recalls)

    # Balanced_accuracy adjusted
    # - Results are adjusted for change
    # - Random scores score 0
    # - Values can be negative
    random_classifier_accuracy = 1 / n_categories
    scaling_factor = 1 - random_classifier_accuracy
    balanced_accuracy_adjusted = (
        balanced_accuracy - random_classifier_accuracy
    ) / scaling_factor

    ### scikitlearn
    # Matthews Correlation Coefficient (MCC)
    # --> https://github.com/scikit-learn/scikit-learn/blob/d99b728b3/sklearn/metrics/_classification.py#L890
    # --> https://dwbi1.wordpress.com/2022/10/05/mcc-formula-for-multiclass-classification/
    # - https://github.com/Lightning-AI/torchmetrics/blob/v1.1.0/src/torchmetrics/functional/classification/matthews_corrcoef.py#L37

    ### xskillscore
    # Hanssen-Kuipers Discriminant / Peirce_score / True skill statistic

    # heidke_score / Cohen’s Kappa (HSS)
    # - Scores above .8 are generally considered good agreement;
    # - Zero or lower means no agreement

    # Cohen Kappa and MCC
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0222916

    # Define metrics
    dictionary = {
        "N": N,
        "N_correct": N_correct,
        "N_wrong": N_wrong,
        "error_rate": error_rate,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_adjusted": balanced_accuracy_adjusted,
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
        "error_rate",
        "accuracy",
        "balanced_accuracy",
        "balanced_accuracy_adjusted",
    ]
    return func, skill_names


@print_elapsed_time(task="deterministic categorical")
def _xr_apply_routine(
    pred,
    obs,
    sample_dims,
    n_categories,
    metrics=None,
    compute=True,
    drop_options=None,
):
    # Retrieve function and skill names
    func, skill_names = get_metrics_info()

    # Define kwargs
    kwargs = {}
    kwargs["drop_options"] = drop_options
    kwargs["n_categories"] = n_categories

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

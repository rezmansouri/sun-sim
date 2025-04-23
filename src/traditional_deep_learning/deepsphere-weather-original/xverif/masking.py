#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:44:07 2023.

@author: ghiggi
"""
import copy
import warnings

import numpy as np


def _get_function_keys():
    keys = [
        "nan",
        "inf",
        "equal_values",
        "values",
        "above_threshold",
        "below_threshold",
    ]
    return keys


def _check_conditioned_on(conditioned_on):
    if conditioned_on not in ["obs", "pred", "any", "both"]:
        raise ValueError("conditioned_on must be one of 'obs', 'pred', 'any', 'both'.")
    return conditioned_on


def _check_masking_option_keys(masking_option):
    keys = list(masking_option.keys())
    if len(keys) > 2:
        raise ValueError("A masking option must contain at most two keys.")
    if len(keys) == 2:
        if "conditioned_on" not in keys:
            raise ValueError(
                "A masking option with two keys must contain 'conditioned_on' as one of the keys."
            )
    valid_keys = [
        "nan",
        "inf",
        "equal_values",
        "values",
        "above_threshold",
        "below_threshold",
        "conditioned_on",
    ]
    unvalid_keys = [key for key in keys if key not in valid_keys]
    if len(unvalid_keys) > 0:
        raise ValueError(
            f"Invalid masking option keys: {unvalid_keys}. Valid keys are {valid_keys}."
        )


def _check_masking_option_values(masking_option):
    valid_values_dict = {
        "nan": [True, False],
        "inf": [True, False],
        "equal_values": [True, False],
        "values": (int, float, list, np.ndarray),
        "above_threshold": (int, float),
        "below_threshold": (int, float),
        "conditioned_on": ["obs", "pred", "any", "both"],
    }
    for key, value in masking_option.items():
        if key in ["nan", "inf", "equal_values", "conditioned_on"]:
            if value not in valid_values_dict[key]:
                valid_values = valid_values_dict[key]
                raise ValueError(
                    f"Invalid masking option '{key}' key value. Valid values are {valid_values}."
                )
        elif key in ["above_threshold", "below_threshold", "values"]:
            if not isinstance(value, valid_values_dict[key]):
                valid_values = valid_values_dict[key]
                raise TypeError(
                    f"Invalid type for masking option '{key}' key value'. Valid types are {valid_values}."
                )


def _set_default_conditioned_on_value(masking_options):
    default_values = {
        "nan": "any",
        "inf": "any",
        "equal_values": "both",  # dummy
        "values": "both",
        "above_threshold": "both",
        "below_threshold": "both",
    }
    for i, masking_option in enumerate(masking_options):
        if "conditioned_on" not in masking_option:
            key = list(masking_option)[0]
            masking_options[i]["conditioned_on"] = default_values[key]
    return masking_options


def _check_masking_option(masking_option):
    _check_masking_option_keys(masking_option)
    _check_masking_option_values(masking_option)


def is_valid_masking_option(masking_option):
    """Check argument validity of a masking option."""
    try:
        _check_masking_option(masking_option)
        validity = True
    except Exception:
        validity = False
    return validity


def check_masking_options(masking_options):
    """Check validity of DataArray masking options list."""
    if isinstance(masking_options, type(None)):
        return []
    if isinstance(masking_options, dict):
        if is_valid_masking_option(masking_options):
            masking_options = [masking_options]
        else:
            raise TypeError("Specify masking_options as a list of dictionaries.")
    if not isinstance(masking_options, list):
        raise TypeError("masking_options must be a list of dictionaries.")
    for masking_option in masking_options:
        _check_masking_option(masking_option)
    masking_options = _set_default_conditioned_on_value(masking_options)
    return masking_options


def _get_masking_option_func_key(masking_option):
    keys = list(masking_option)
    valid_func_keys = _get_function_keys()
    keys = [key for key in keys if key in valid_func_keys]
    if len(keys) != 1:
        raise ValueError(
            f"A masking option must contain only one of the following keys: {valid_func_keys}'."
        )
    return keys[0]


class MaskDataArrays:
    """MaskDataArray class."""

    def __init__(self, pred, obs, masking_options=None, masking_value=np.nan):
        """Initialize the Masking Object."""
        masking_options = check_masking_options(masking_options)
        # TODO: check masking_value same type of pred and obs
        self.pred = pred
        self.obs = obs
        self.masked_pred = self.pred.copy()
        self.masked_obs = self.obs.copy()
        self.masking_value = masking_value
        self.masking_options = masking_options

    def apply(self):
        """Apply the masking options."""
        func_dict = {
            "nan": self.nan,
            "inf": self.inf,
            "equal_values": self.equal_values,
            "values": self.values,
            "above_threshold": self.above_threshold,
            "below_threshold": self.below_threshold,
        }

        for masking_option in self.masking_options:
            option_kwargs = masking_option.copy()
            func_key = _get_masking_option_func_key(masking_option)
            func = func_dict[func_key]

            if func_key in ["nan", "inf", "equal_values"]:
                if not masking_option[func_key]:
                    continue  # do not execute masking if False
                option_kwargs.pop(func_key, None)
            if func_key == "above_threshold":
                option_kwargs["threshold"] = option_kwargs.pop("above_threshold")
            if func_key == "below_threshold":
                option_kwargs["threshold"] = option_kwargs.pop("below_threshold")
            if "masking_value" not in option_kwargs:
                option_kwargs["masking_value"] = self.masking_value

            # Apply masking
            self.masked_pred, self.masked_obs = func(**option_kwargs)

        return self.masked_pred, self.masked_obs

    def nan(self, masking_value=np.nan, conditioned_on="any"):
        """Mask nan values."""
        pred = self.masked_pred
        obs = self.masked_obs
        condition_map = {
            "obs": lambda: np.isnan(obs),
            "pred": lambda: np.isnan(pred),
            "any": lambda: np.logical_or(np.isnan(obs), np.isnan(pred)),
            "both": lambda: np.logical_and(np.isnan(obs), np.isnan(pred)),
        }
        isnan = condition_map[conditioned_on]()
        pred = pred.where(~isnan, other=masking_value)
        obs = obs.where(~isnan, other=masking_value)
        return pred, obs

    def inf(self, masking_value=np.nan, conditioned_on="any"):
        """Mask inf values."""
        pred = self.masked_pred
        obs = self.masked_obs
        condition_map = {
            "obs": lambda: np.isinf(obs),
            "pred": lambda: np.isinf(pred),
            "any": lambda: np.logical_or(np.isinf(obs), np.isinf(pred)),
            "both": lambda: np.logical_and(np.isinf(obs), np.isinf(pred)),
        }
        isinf = condition_map[conditioned_on]()
        pred = pred.where(~isinf, other=masking_value)
        obs = obs.where(~isinf, other=masking_value)
        return pred, obs

    def equal_values(self, masking_value=np.nan, conditioned_on="dummy"):
        """Mask the values which are equal in both DataArrays."""
        pred = self.masked_pred
        obs = self.masked_obs
        isequal = pred == obs
        pred = pred.where(~isequal, other=masking_value)
        obs = obs.where(~isequal, other=masking_value)
        return pred, obs

    def values(self, values, masking_value=np.nan, conditioned_on="both"):
        """Mask the specified values."""
        pred = self.masked_pred
        obs = self.masked_obs
        if isinstance(values, (int, float)):
            values = [values]
        condition_map = {
            "obs": lambda: np.isin(obs, values),
            "pred": lambda: np.isin(pred, values),
            "any": lambda: np.logical_or(np.isin(obs, values), np.isin(pred, values)),
            "both": lambda: np.logical_and(np.isin(obs, values), np.isin(pred, values)),
        }
        isvalues = condition_map[conditioned_on]()
        pred = pred.where(~isvalues, other=masking_value)
        obs = obs.where(~isvalues, other=masking_value)
        return pred, obs

    def above_threshold(self, threshold, masking_value=np.nan, conditioned_on="both"):
        """Mask values above the specified threshold."""
        pred = self.masked_pred
        obs = self.masked_obs
        condition_map = {
            "obs": lambda: obs > threshold,
            "pred": lambda: pred > threshold,
            "any": lambda: np.logical_or(obs > threshold, pred > threshold),
            "both": lambda: np.logical_and(obs > threshold, pred > threshold),
        }
        isabove = condition_map[conditioned_on]()
        pred = pred.where(~isabove, other=masking_value)
        obs = obs.where(~isabove, other=masking_value)
        return pred, obs

    def below_threshold(self, threshold, masking_value=np.nan, conditioned_on="both"):
        """Mask values below the specified threshold."""
        pred = self.masked_pred
        obs = self.masked_obs
        condition_map = {
            "obs": lambda: obs < threshold,
            "pred": lambda: pred < threshold,
            "any": lambda: np.logical_or(obs < threshold, pred < threshold),
            "both": lambda: np.logical_and(obs < threshold, pred < threshold),
        }
        isbelow = condition_map[conditioned_on]()
        pred = pred.where(~isbelow, other=masking_value)
        obs = obs.where(~isbelow, other=masking_value)
        return pred, obs


####--------------------------------------------------------------------------.
#### Wrappers


def mask_dataarrays(pred, obs, masking_options):
    """Mask a DataArray based on the given masking options."""
    pred, obs = MaskDataArrays(pred, obs, masking_options=masking_options).apply()
    return pred, obs


def mask_datasets(pred, obs, masking_options):
    """Mask datasets according to masking_options.

    Accepted masking_options format:
    --> {var1: {masking_option}, var2: {masking_option}}
    --> [{'nan': True}, {'values': 0, conditioned_on='both'})

    """
    masking_options = _check_per_dataset_masking_options(pred, masking_options)
    for var, var_masking_options in masking_options.items():
        masked_pred, masked_obs = mask_dataarrays(
            pred[var], obs[var], masking_options=var_masking_options
        )
        pred[var] = masked_pred
        obs[var] = masked_obs
    return pred, obs


def _is_per_variable_masking_options(pred, masking_options):
    """Check if the masking_options is specific to Dataset variables."""
    if isinstance(masking_options, dict):
        valid_variables = list(pred.data_vars)
        unvalid_keys = [
            key for key in masking_options.keys() if key not in valid_variables
        ]
        if len(unvalid_keys) == 0 and len(masking_options) != 0:
            specified_variables = list(masking_options.keys())
            if len(specified_variables) != len(valid_variables):
                unspecified_variables = [
                    var for var in valid_variables if var not in specified_variables
                ]
                warnings.warn(
                    f"No masking options specified for {unspecified_variables} variables.",
                    stacklevel=3,
                )  # TODO: xverif warning class
            return True
        else:
            return False
    return False


def _check_per_dataset_masking_options(pred, masking_options):
    """Check and ensure that the masking_options is defined per Dataset variable."""
    if _is_per_variable_masking_options(pred, masking_options):
        for var, var_masking_options in masking_options.items():
            masking_options[var] = check_masking_options(var_masking_options)
    else:
        masking_options = check_masking_options(masking_options)
        masking_options = {
            var: copy.deepcopy(masking_options) for var in list(pred.data_vars)
        }
    return masking_options

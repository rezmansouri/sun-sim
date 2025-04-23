#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:46:45 2023.

@author: ghiggi
"""
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


def _check_dropping_option_keys(dropping_option):
    keys = list(dropping_option.keys())
    if len(keys) > 2:
        raise ValueError("A dropping option must contain at most two keys.")
    if len(keys) == 2:
        if "conditioned_on" not in keys:
            raise ValueError(
                "A dropping option with two keys must contain 'conditioned_on' as one of the keys."
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
            f"Invalid dropping option keys: {unvalid_keys}. Valid keys are {valid_keys}."
        )


def _check_dropping_option_values(dropping_option):
    valid_values_dict = {
        "nan": [True, False],
        "inf": [True, False],
        "equal_values": [True, False],
        "values": (int, float, list, np.ndarray),
        "above_threshold": (int, float),
        "below_threshold": (int, float),
        "conditioned_on": ["obs", "pred", "any", "both"],
    }
    for key, value in dropping_option.items():
        if key in ["nan", "inf", "equal_values", "conditioned_on"]:
            if value not in valid_values_dict[key]:
                valid_values = valid_values_dict[key]
                raise ValueError(
                    f"Invalid dropping option '{key}' key value. Valid values are {valid_values}."
                )
        elif key in ["above_threshold", "below_threshold", "values"]:
            if not isinstance(value, valid_values_dict[key]):
                valid_values = valid_values_dict[key]
                raise TypeError(
                    f"Invalid type for dropping option '{key}' key value'. Valid types are {valid_values}."
                )


def _set_default_conditioned_on_value(drop_options):
    default_values = {
        "nan": "any",
        "inf": "any",
        "equal_values": "both",  # dummy
        "values": "both",
        "above_threshold": "both",
        "below_threshold": "both",
    }
    for i, dropping_option in enumerate(drop_options):
        if "conditioned_on" not in dropping_option:
            key = list(dropping_option)[0]
            drop_options[i]["conditioned_on"] = default_values[key]
    return drop_options


def _check_dropping_option(dropping_option):
    _check_dropping_option_keys(dropping_option)
    _check_dropping_option_values(dropping_option)


def is_valid_dropping_option(dropping_option):
    """Check argument validity of a dropping option."""
    try:
        _check_dropping_option(dropping_option)
        validity = True
    except Exception:
        validity = False
    return validity


def check_drop_options(drop_options):
    """Check validity of DataArray dropping options list."""
    if isinstance(drop_options, type(None)):
        return []
    if isinstance(drop_options, dict):
        if is_valid_dropping_option(drop_options):
            drop_options = [drop_options]
        else:
            raise TypeError("Specify drop_options as a list of dictionaries.")
    if not isinstance(drop_options, list):
        raise TypeError("drop_options must be a list of dictionaries.")
    for dropping_option in drop_options:
        _check_dropping_option(dropping_option)
    drop_options = _set_default_conditioned_on_value(drop_options)
    return drop_options


def _get_dropping_option_func_key(dropping_option):
    keys = list(dropping_option)
    valid_func_keys = _get_function_keys()
    keys = [key for key in keys if key in valid_func_keys]
    if len(keys) != 1:
        raise ValueError(
            f"A dropping option must contain only one of the following keys: {valid_func_keys}'."
        )
    return keys[0]


class DropData:
    """DropData class."""

    def __init__(self, pred, obs, drop_options=None):
        """Initialize the object."""
        if pred.shape != obs.shape:
            raise ValueError("'pred' and 'obs' must have same shape.")
        if pred.ndim > 1:
            raise ValueError("Expecting 1D arrays.")
        drop_options = check_drop_options(drop_options)
        self.pred = pred
        self.obs = obs
        self.dropped_pred = self.pred.copy()
        self.dropped_obs = self.obs.copy()
        self.drop_options = drop_options

    def apply(self):
        """Apply the dropping options."""
        func_dict = {
            "nan": self.nan,
            "inf": self.inf,
            "equal_values": self.equal_values,
            "values": self.values,
            "above_threshold": self.above_threshold,
            "below_threshold": self.below_threshold,
        }

        for dropping_option in self.drop_options:
            option_kwargs = dropping_option.copy()
            func_key = _get_dropping_option_func_key(dropping_option)
            func = func_dict[func_key]

            if func_key in ["nan", "inf", "equal_values"]:
                if not dropping_option[func_key]:
                    continue  # do not execute masking if False
                option_kwargs.pop(func_key, None)
            if func_key == "above_threshold":
                option_kwargs["threshold"] = option_kwargs.pop("above_threshold")
            if func_key == "below_threshold":
                option_kwargs["threshold"] = option_kwargs.pop("below_threshold")

            # Apply masking
            self.dropped_pred, self.dropped_obs = func(**option_kwargs)

        return self.dropped_pred, self.dropped_obs

    def nan(self, conditioned_on="any"):
        """Mask nan values."""
        pred = self.dropped_pred
        obs = self.dropped_obs
        condition_map = {
            "obs": lambda: np.isnan(obs),
            "pred": lambda: np.isnan(pred),
            "any": lambda: np.logical_or(np.isnan(obs), np.isnan(pred)),
            "both": lambda: np.logical_and(np.isnan(obs), np.isnan(pred)),
        }
        isnan = condition_map[conditioned_on]()
        pred = pred[~isnan]
        obs = obs[~isnan]
        return pred, obs

    def inf(self, conditioned_on="any"):
        """Mask inf values."""
        pred = self.dropped_pred
        obs = self.dropped_obs
        condition_map = {
            "obs": lambda: np.isinf(obs),
            "pred": lambda: np.isinf(pred),
            "any": lambda: np.logical_or(np.isinf(obs), np.isinf(pred)),
            "both": lambda: np.logical_and(np.isinf(obs), np.isinf(pred)),
        }
        isinf = condition_map[conditioned_on]()
        pred = pred[~isinf]
        obs = obs[~isinf]
        return pred, obs

    def equal_values(self, conditioned_on="dummy"):
        """Mask the values which are equal in both DataArrays."""
        pred = self.dropped_pred
        obs = self.dropped_obs
        isequal = pred == obs
        pred = pred[~isequal]
        obs = obs[~isequal]
        return pred, obs

    def values(self, values, conditioned_on="both"):
        """Mask the specified values."""
        pred = self.dropped_pred
        obs = self.dropped_obs
        if isinstance(values, (int, float)):
            values = [values]
        condition_map = {
            "obs": lambda: np.isin(obs, values),
            "pred": lambda: np.isin(pred, values),
            "any": lambda: np.logical_or(np.isin(obs, values), np.isin(pred, values)),
            "both": lambda: np.logical_and(np.isin(obs, values), np.isin(pred, values)),
        }
        isvalues = condition_map[conditioned_on]()
        pred = pred[~isvalues]
        obs = obs[~isvalues]
        return pred, obs

    def above_threshold(self, threshold, conditioned_on="both"):
        """Mask values above the specified threshold."""
        pred = self.dropped_pred
        obs = self.dropped_obs
        condition_map = {
            "obs": lambda: obs > threshold,
            "pred": lambda: pred > threshold,
            "any": lambda: np.logical_or(obs > threshold, pred > threshold),
            "both": lambda: np.logical_and(obs > threshold, pred > threshold),
        }
        isabove = condition_map[conditioned_on]()
        pred = pred[~isabove]
        obs = obs[~isabove]
        return pred, obs

    def below_threshold(self, threshold, conditioned_on="both"):
        """Mask values below the specified threshold."""
        pred = self.dropped_pred
        obs = self.dropped_obs
        condition_map = {
            "obs": lambda: obs < threshold,
            "pred": lambda: pred < threshold,
            "any": lambda: np.logical_or(obs < threshold, pred < threshold),
            "both": lambda: np.logical_and(obs < threshold, pred < threshold),
        }
        isbelow = condition_map[conditioned_on]()
        pred = pred[~isbelow]
        obs = obs[~isbelow]
        return pred, obs


####--------------------------------------------------------------------------.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:57:49 2022

@author: ghiggi
"""
import xarray
from .checks import check_valid_scaler

# -----------------------------------------------------------------------------.
# #########################
#### SequentialScalers ####
# #########################
# SequentialScaler(scaler1, scaler2, ... )
# --> (PixelwiseAnomalies + GlobalStandardScaler)
# --> (Ad-hoc scaler for specific variables)
# --> Allow nested SequentialScaler


class SequentialScaler:
    """Enable sequential scaling operations."""

    def __init__(self, *scalers):
        # Check is a valid scaler
        for scaler in scalers:
            check_valid_scaler(scaler)
        self.list_scalers = scalers
        self.fitted = False
        self.scaler_class = "SequentialScaler"

    def fit(self, show_progress=True):
        """Fit all scalers within a SequentialScaler."""
        new_list_scaler = []
        for scaler in self.list_scalers:
            if not scaler.fitted:
                scaler.fit(show_progress=show_progress)
            new_list_scaler.append(scaler)
        self.list_scalers = new_list_scaler
        self.fitted = True

    def save(self):
        """Save a SequentialScaler to disk."""
        raise NotImplementedError(
            "Saving of SequentialScaler has not been yet implemented!"
        )

    def transform(self, new_data, variable_dim=None, rename_dict=None):
        """Transform data using the fitted SequentialScaler."""
        for scaler in self.list_scalers:
            if not scaler.fitted:
                raise ValueError(
                    "The SequentialScaler contains scalers that have not been fit. Use .fit() first!"
                )
        for scaler in self.list_scalers:
            new_data = scaler.transform(
                new_data=new_data, variable_dim=variable_dim, rename_dict=rename_dict
            )
        return new_data

    def inverse_transform(self, new_data, variable_dim=None, rename_dict=None):
        """Inverse transform data using the fitted SequentialScaler."""
        reversed_scalers = self.list_scalers[::-1]
        for scaler in reversed_scalers:
            if not scaler.fitted:
                raise ValueError(
                    "The SequentialScaler contains scalers that have not been fit. Use .fit() first!"
                )
        for scaler in reversed_scalers:
            new_data = scaler.inverse_transform(
                new_data=new_data, variable_dim=variable_dim, rename_dict=rename_dict
            )
        return new_data

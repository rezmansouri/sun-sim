#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:53:44 2022

@author: ghiggi
"""
from .temporal_scalers import TemporalStandardScaler


class AnomalyScaler(TemporalStandardScaler):
    """Class object to transform data into anomalies (and back)."""

    def __init__(
        self,
        data,
        time_dim,
        time_groups=None,
        variable_dim=None,
        groupby_dims=None,
        reference_period=None,
        standardized=True,
        eps=0.0001,
        ds_anomaly=None,
    ):
        super().__init__(
            data=data,
            time_dim=time_dim,
            time_groups=time_groups,
            variable_dim=variable_dim,
            groupby_dims=groupby_dims,
            reference_period=reference_period,
            center=True,
            standardize=True,
            eps=eps,
            ds_scaler=ds_anomaly,
        )
        # Set default method
        self.standardize = standardized

    def transform(
        self, new_data, standardized=None, variable_dim=None, rename_dict=None
    ):
        """Transform new_data to anomalies."""
        # Standardize option
        standardize_default = self.standardize
        if standardized is not None:
            self.standardize = standardized
        # Get anomalies
        anom = TemporalStandardScaler.transform(
            self, new_data=new_data, variable_dim=variable_dim, rename_dict=rename_dict
        )
        # Reset default
        self.standardize = standardize_default
        return anom

    def inverse_transform(
        self, new_data, standardized=None, variable_dim=None, rename_dict=None
    ):
        """Retrieve original values from anomalies."""
        # Standardize option
        standardize_default = self.standardize
        if standardized is not None:
            self.standardize = standardized
        # Inverse
        x = TemporalStandardScaler.inverse_transform(
            self, new_data=new_data, variable_dim=variable_dim, rename_dict=rename_dict
        )
        # Reset default
        self.standardize = standardize_default
        return x

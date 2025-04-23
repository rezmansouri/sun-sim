#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:39:32 2023.

@author: ghiggi
"""
import warnings
from functools import wraps


def suppress_warnings(func):
    """Decorator to suppress warnings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            # Specify the warnings to filter out here
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # warnings.filterwarnings("ignore", msg="custom warning")
            return func(*args, **kwargs)

    return wrapper

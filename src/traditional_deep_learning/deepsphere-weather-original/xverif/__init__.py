#!/usr/bin/env python3
"""
Created on Thu Mar  3 17:05:50 2022.

@author: ghiggi
"""
from xverif.wrappers import deterministic

__all__ = [
    "deterministic",
    "probabilistic",
    "spatial",
]

# Arbitrary small yet strictly positive number to avoid undefined results when
# dividing by zero
EPS = 0.000001  # epsilon = np.finfo(np.float64).eps

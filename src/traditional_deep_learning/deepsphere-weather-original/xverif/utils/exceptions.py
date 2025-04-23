#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:29:40 2023.

@author: ghiggi
"""


class MissingOptionalDependency(Exception):
    """Raised when an optional dependency is needed but not found."""

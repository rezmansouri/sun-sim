#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:53:01 2023.

@author: ghiggi
"""
import datetime
from time import perf_counter


def print_elapsed_time(task=""):
    """Decorator which print the execution time of a task."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = perf_counter()
            results = func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            timedelta_str = str(datetime.timedelta(seconds=execution_time))
            print(f" Elapsed time for {task} verification: {timedelta_str} .", end="\n")
            return results

        return wrapper

    return decorator

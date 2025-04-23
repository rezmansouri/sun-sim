#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:45:52 2023.

@author: ghiggi
"""
# If variable have different number of classes
# --> Need to loop over variable and specify separate n_categories !

# Overall Accuracy Metrics
# - Confusion Matrix and Derived Metrics


# Class-specific Metrics
# --> One against the other --> All binary metrics
# --> Output contains the 'category' dimension

# - multiclass_mode="one-vs-all", "one-vs-one
# one-vs-all (OvR) - computes the average of the scores for each class against all other classes (macro/weighted/micro)
# one-vs-one (OvO) - computes the average of all possible pairwise combinations of classes  (macro/weighted)


# Aggregate Metrics Across Classes
# - Micro-average: Aggregate the contributions of all classes to compute the average metric.
# - Macro-average: Compute the metric independently for each class and then take the average, treating all classes equally
# - Weighted average: Take into account the class imbalance by computing the average of metrics, weighted by the number of true instances for each class.
# --> Output contains the 'averaging_method'  dimension
# --> Derived from the class-specific metrics
# --> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn-metrics-f1-score

# averaging_method
# 'macro':
# Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

# 'weighted':
# Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).

# 'micro':
# Calculate metrics globally by counting the sum of TP, TN, FP, FN  (for each class) and then compute the statistics.

# --> Micro deals around the confusion matrix
# --> Macro and weighted can be vectorized (average of class-specific metrics)

# --> https://github.com/scikit-learn/scikit-learn/blob/d99b728b3a7952b2111cf5e0cb5d14f92c6f3a80/sklearn/metrics/_classification.py#L1551

# --> https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f

####--------------------------------------------------------------------------.
# https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
# https://www.evidentlyai.com/classification-metrics/multi-class-metrics

# sample_weights
# --> _weighted_sum, _weighted_mean


# Multiclass model has one target feature, and more-than-2 (mutually exclusive) outcomes.

# Multilabel model has multiple target features, but each has a binary outcome (0 or 1)
# --> Verification of multilabel model not implemented

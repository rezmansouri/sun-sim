#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:13:34 2023.

@author: ghiggi
"""
####--------------------------------------------------------------------------.
#### Ordinal Regression models
# Proportional Odds Assumption Check - POAC
# - Also referred to as the Parallel Lines Assumption
# - Crucial assumption made by ordinal logistic regression models
# - Assumption:
#    - The relationship between each pair of outcome groups is statistically the same.
#    - In other words, the odds of observing an outcome in category j or below vs. above category
#      are proportional regardless of the value of j
# --> Score Test (Brant Test)
# --> Require fitting a logistic regression model
# --> NotImplemented

# Ordinal Regression Threshold Metrics - ORTM
# - Ordinal regression models, especially the cumulative link models (like the proportional odds model),
# - is that they employ "thresholds" or "cut-points" to delineate the boundaries between the ordinal categories.
# - The essence of these models is to predict the cumulative probabilities up to each threshold. For a response variable with
#   k levels, there will be k−1 thresholds. Each threshold defines the cut-off between two consecutive ordinal categories.
# --> Cut thresholds, category probabilities, ...

####--------------------------------------------------------------------------.

# For ordered data, all continuous and multiclass metrics can be used (with caution in the interpretation)!

# Ordered (ordinal) data refers to categorical data with a defined order but without a consistent distance between categories.
# For such data, you'd want metrics that recognize the inherent order of the categories.


# Cumulative Log Odds Ratio:

# This metric looks at the cumulative odds of being in a higher category versus a lower category. It's particularly suited for ordinal categories.
# Mean Absolute Error (Ordinal Version):

# You can treat ordinal categories as numerical rankings (1, 2, 3, ...), and then compute the mean absolute error between the predicted and true rankings. This method considers the order of the categories.
# Proportional Odds Assumption Check:

# While this is more of a diagnostic tool than a metric, checking whether the proportional odds assumption holds in ordinal regression can be useful in some contexts.
# Spearman's Rank Correlation Coefficient:

# Measures the strength and direction of the monotonic relationship between the true and predicted rankings.

# OMAE - Ordinal Mean Absolute Error
# - Simply the MAE over the ordinal data
# - Provide the average ordinal distance
# - OMAE of 0.5 might indicate that, on average, predictions are off by half an ordinal rank
# - Prediction that's off by two ranks is penalized twice as much as a prediction that's off by one rank.

# MacroMAE - Macroaveraged MAE
# - Is the average of per-class MAEs
# - It treats all classes equally, regardless of their frequency
# - Equal weight to all categories, thus nullifying the effects of imbalance.
# - Particularly useful in scenarios where the classes are imbalanced, as it ensures that performance on
#   the smaller classes influences the final metric as much as performance on the larger classes.

# WK (QK) Weighted (Cohen's/Quadratic) Kappa
# - Extension of Cohen's Kappa, with the primary difference being the assignment of weights to each cell in the confusion matrix.
# - Accounts for both complete agreement and partial agreement (based on the order).
# - It provides more credit for predictions that are closer in order to the true class than those that are further away.
# - Gives more penalty to predictions that are further away from the actual values

# - Quadratic weights ensure that predictions further away from the actual value are penalized more
# - The weight matrix is based on the squared differences between categories

# - A QWK value of 1 indicates perfect agreement.
# - A value of 0 indicates the agreement is no better than random.
# - Negative values suggest an agreement worse than random

# Cumulative Log Odds Ratio - CLOR
# - Measure the difference between the observed cumulative log odds and the predicted cumulative log odds.
# - Captures the discrepancies between the model's predictions and the actual data distribution across the ordinal categories
# - A value of 0 indicates that the predicted log odds match the observed log odds perfectly for all ordinal levels.
# - High values sggest a discrepancy between the model's predictions and the actual distribution of the ordinal categories.

# - The odds of the event occurring is the probability of the event divided by the probability of the event not occurring
# - The cumulative odd of a event is the defined as the odds of observing an outcome up to and including category
#   j versus observing an outcome in a category higher than j.
# - The CLOR is computed as the sum of the squared difference (for each category) of the predicted and observed cumulate log odds.


# # Example
# obs = np.array([0, 1, 2, 1, 0, 2])
# pred = np.array([0, 1, 1, 0, 0, 2])
# print(macro_mae(obs, pred))

# import numpy as np

# ### Inputs
# # Expects from 0 to n_categories

# #### Optional input arguments
# # n_categories
# # misclassification_weights

# # Add utility to return only the confusion matrix


# def multiclass_confusion_matrix(obs, pred, n_categories=None):
#     """Compute the multiclass confusion matrix."""
#     if n_categories is None:
#         n_categories = max(np.max(obs), np.max(pred)) + 1
#     return np.histogram2d(obs, pred, bins=(n_categories, n_categories))[0]


# def weighted_error_rate(conf_matrix, weights=None):
#     """Compute the weighted error rate from a confusion matrix."""
#     if weights is None:
#         weights = np.ones_like(conf_matrix)
#     total_samples = np.sum(conf_matrix)
#     weighted_errors = np.sum(weights * conf_matrix)
#     # Subtract diagonal since those aren't errors
#     weighted_errors -= np.sum(np.diag(weights) * np.diag(conf_matrix))
#     return weighted_errors / total_samples


# def weighted_accuracy(conf_matrix, weights=None):
#     """Compute the weighted accuracy from a confusion matrix."""
#     return 1 - weighted_error_rate(conf_matrix, weights)


# def misclassification_spread(conf_matrix):
#     """Compute the misclassification spread from a confusion matrix."""
#     total_misclassifications = np.sum(conf_matrix) - np.sum(np.diag(conf_matrix))
#     rows, cols = np.indices(conf_matrix.shape)
#     spread = np.sum(np.abs(rows - cols) * conf_matrix)
#     return spread / total_misclassifications


# def overall_error_rate(conf_matrix):
#     """Compute the overall error rate from a confusion matrix."""
#     total_samples = np.sum(conf_matrix)
#     correct_predictions = np.sum(np.diag(conf_matrix))
#     return (total_samples - correct_predictions) / total_samples


# def gerrity_score(y_true, y_pred, n_categories):
#     # --> https://rdrr.io/github/joaofgoncalves/SegOptim/man/GerritySkillScore.html
#     # --> https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/contingency.py#L735
#     # --> https://www.ecmwf.int/sites/default/files/elibrary/2010/11988-new-equitable-score-suitable-verifying-precipitation-nwp.pdf

#     # CHATGPT TRIAL

#     # Create weights matrix based on squared differences of ranks
#     # ranks = np.arange(n_categories)
#     # weights_matrix = 1 - ((ranks[:, None] - ranks) ** 2) / (n_categories - 1)**2

#     weights_matrix = 1 - np.abs(
#         np.arange(n_categories)[:, None] - np.arange(n_categories)
#     ) / (n_categories - 1)

#     # Compute confusion matrix
#     hist2d, _, _ = np.histogram2d(
#         y_true,
#         y_pred,
#         bins=n_categories,
#         range=[[-0.5, n_categories - 0.5], [-0.5, n_categories - 0.5]],
#     )
#     confusion_matrix = hist2d.T

#     # Compute the Gerrity score
#     score = np.sum(weights_matrix * confusion_matrix) / np.sum(confusion_matrix)

#     return score


# # Example usage
# obs = np.array([0, 1, 2, 1, 0, 2, 2, 0, 1])
# pred = np.array([0, 1, 1, 0, 0, 2, 2, 1, 2])

# # Compute confusion matrix
# conf_matrix = multiclass_confusion_matrix(obs, pred)

# # Weights for misclassification (for demonstration; default is all ones)
# weights = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

# print("Confusion Matrix:\n", conf_matrix)
# print("Weighted Error Rate:", weighted_error_rate(conf_matrix, weights))
# print("Weighted Accuracy:", weighted_accuracy(conf_matrix, weights))
# print("Misclassification Spread:", misclassification_spread(conf_matrix))
# print("Overall Error Rate:", overall_error_rate(conf_matrix))


# def macro_mae(obs, pred):
#     """
#     Compute the Macroaverage Mean Absolute Error.

#     Parameters
#     ----------
#     - obs: numpy array of true labels (should be integer labels from 0 to C-1 where C is the number of classes)
#     - pred: numpy array of predicted values (should be in the same format as obs)

#     Returns
#     -------
#     - Macro-MAE value
#     """
#     # List to store the MAE of each class
#     maes = []

#     # Unique classes in the observation
#     classes = np.unique(obs)

#     # Compute MAE for each class and append to the list
#     for c in classes:
#         # Binary mask for the current class
#         mask = obs == c
#         # Compute and store the MAE for the current class
#         maes.append(np.mean(np.abs(pred[mask] - c)))

#     # Return the average of the per-class MAEs
#     return np.mean(maes)


# def macro_mae(obs, pred):
#     """
#     Compute the Macroaverage Mean Absolute Error using vectorized operations.

#     Parameters
#     ----------
#     - obs: numpy array of true labels
#     - pred: numpy array of predicted values

#     Returns
#     -------
#     - Macro-MAE value
#     """
#     # Get unique classes and their counts
#     classes, counts = np.unique(obs, return_counts=True)

#     # Calculate absolute errors for each class using broadcasting
#     abs_errors = np.abs(pred[:, np.newaxis] - classes)

#     # Mask out errors for classes other than the true class
#     masked_errors = np.where(
#         abs_errors == abs_errors.min(axis=1, keepdims=True), abs_errors, 0
#     )

#     # Sum errors for each class and then average
#     classwise_errors = masked_errors.sum(axis=0) / counts

#     return classwise_errors.mean()


# import numpy as np


# def weighted_kappa_vectorized(obs, pred):
#     classes = np.unique(np.concatenate((obs, pred)))
#     conf_matrix = np.histogram2d(obs, pred, bins=(classes, classes))[0]

#     weights = np.arange(len(classes))[:, None] - np.arange(len(classes))
#     weights = weights**2

#     obs_sum = np.sum(conf_matrix, axis=1)
#     pred_sum = np.sum(conf_matrix, axis=0)
#     expected = np.outer(obs_sum, pred_sum) / len(obs)

#     weighted_observed = np.sum(weights * conf_matrix)
#     weighted_expected = np.sum(weights * expected)

#     return 1 - (weighted_observed / weighted_expected)


# import numpy as np


# def weighted_kappa(obs, pred):
#     # Get unique classes and the number of classes
#     classes = np.unique(np.concatenate((obs, pred)))
#     n_categories = len(classes)

#     # Confusion matrix
#     conf_matrix = np.zeros((n_categories, n_categories))
#     for i in range(len(obs)):
#         conf_matrix[obs[i]][pred[i]] += 1

#     # Weights
#     weights = np.zeros((n_categories, n_categories))
#     for i in range(n_categories):
#         for j in range(n_categories):
#             weights[i][j] = (i - j) ** 2

#     # Expected matrix under independence
#     obs_sum = np.sum(conf_matrix, axis=1)
#     pred_sum = np.sum(conf_matrix, axis=0)
#     expected = np.outer(obs_sum, pred_sum) / len(obs)

#     # Weighted observed and expected
#     weighted_observed = np.sum(weights * conf_matrix)
#     weighted_expected = np.sum(weights * expected)

#     return 1 - (weighted_observed / weighted_expected)


# import numpy as np


# def cumulative_log_odds_ratio(obs, pred):
#     """
#     Compute the Cumulative Log Odds Ratio (C-LOR) for ordinal data.

#     For each ordinal level k, compute the cumulative odds as the ratio of the number of observations less than or equal to k to the number greater than k.
#     Take the natural log of the odds ratio to get the log odds.
#     C-LOR is then the sum of the squared differences between the observed and predicted log odds.

#     Parameters
#     ----------
#     - obs: numpy array of true ordinal labels
#     - pred: numpy array of predicted ordinal labels

#     Returns
#     -------
#     - C-LOR value
#     """
#     classes = np.unique(np.concatenate((obs, pred)))

#     def compute_log_odds(data):
#         less_than_or_equal_to_k = np.array([np.sum(data <= k) for k in classes])
#         greater_than_k = len(data) - less_than_or_equal_to_k
#         odds = less_than_or_equal_to_k / greater_than_k
#         return np.log(odds + np.finfo(float).eps)  # small value added to prevent log(0)

#     log_odds_obs = compute_log_odds(obs)
#     log_odds_pred = compute_log_odds(pred)

#     lor = np.sum((log_odds_obs - log_odds_pred) ** 2)

#     return lor


# # Example
# obs = np.array([0, 1, 2, 1, 0, 2])
# pred = np.array([0, 1, 1, 0, 0, 2])
# print(cumulative_log_odds_ratio(obs, pred))


# Add spearmann corr
# Add pearson corr

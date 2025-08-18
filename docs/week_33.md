---
layout: default
title: New loss functions
---

Paper 1 and 2 loss functions:

$$
\text{L}_2^{(2D)} = \frac{1}{BCR} \sum_{b=1}^{B} \sum_{c=1}^{C} \sum_{r=1}^{R}
\left( \sum_{i=1}^{H} \sum_{j=1}^{W} \left| x_{bcrij} - y_{bcrij} \right|^2 \right)^{1/2}
$$

Caused the SFNO to not match the rotation of the sun. Regardless of the pattern.

The new loss function should match the whole datacube with the ground truth at once; i.e., no means across dims etc.

## Metrics that we've seen -> Differentiable and Transformed into Distance Functions:

### 1. Earth Mover's Distance

- Current open-source implementation is 2D
- I took the mean across the $$r$$ dim for reports
- 3D implementation required
- Some implementations for 3D point cloud etc:
    - [PyTorch EMDLoss](https://github.com/meder411/PyTorch-EMDLoss)
    - [pointGAN EMD](https://github.com/fxia22/pointGAN/tree/74b6c432c5eaa1e0a833e755f450df2ee2c5488e/emd)
    - [SciPy 2D](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)
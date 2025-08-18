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

$$
W_1(P, Q) \;=\; \int_{-\infty}^{\infty} \big| F_P(x) - F_Q(x) \big| \, dx
$$

- **$$W_1(P, Q)$$**  
  The Earth Mover’s Distance between two 1D probability distributions \(P\) and \(Q\).

- **$$F_P(x), F_Q(x)$$**  
  The cumulative distribution functions (CDFs) of $$P$$ and $$Q$$:  
  $$
  F_P(x) = \int_{-\infty}^{x} p(t) \, dt, 
  \quad 
  F_Q(x) = \int_{-\infty}^{x} q(t) \, dt
  $$

- **$$|F_P(x) - F_Q(x)|$$**  
  At position $$x$$, this represents the imbalance in total mass between the two distributions up to that point.

- **The integral**  
  Summing the absolute imbalance over the whole line gives the minimum “work” required to transform $$P$$ into $$Q$$.


- Current open-source implementation is 2D
- I took the mean across the $$r$$ dim for reports
- 3D implementation required
- Some implementations for 3D point cloud etc:
    - [PyTorch EMDLoss](https://github.com/meder411/PyTorch-EMDLoss)
    - [pointGAN EMD](https://github.com/fxia22/pointGAN/tree/74b6c432c5eaa1e0a833e755f450df2ee2c5488e/emd)
    - [SciPy 2D](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)
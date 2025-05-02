---
layout: default
title: Metrics and Ready for Optimal SFNO vs HUX
---
# 1. Regression Performance Metrics

## 1.1. RMSE

Root mean squared error

Lower better

$$
RMSE=\sqrt{MSE(y, \hat{y})}
$$

## 1.2 NSE ($$R^2$$)

Nash-Sutcliffe Efficiency

Higher better

$$
NSE = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - {y}_{clim})^2}
$$

- $${y}_{clim}$$ is the climatology
    - a `(140, 111, 128)` data cube
    - mean of the training data

<img src="resources/week_19/climatology.gif">

- Our NSE will be mean of this score across all coordinates
- Range: $$(-\infty,1]$$
    - $$0$$ means $$\hat{y}={y}_{clim}$$
- NNSE (Normalized NSE)
    - So, $$NNSE = 1 / (2-NSE)$$
    - Range: $$[0,1]$$
    - $$0.5$$ means $$\hat{y}={y}_{clim}$$


## 1.3. ACC

- Higher better
- Anomaly Correlation Coefficient
- Used in SFNO paper for spatiotemporal ERA5 data, weather prediction

$$
ACC = \frac{\sum(\hat{y}-{y}_{clim})(y-{y}_{clim})}{\sqrt{\sum{(\hat{y}-{y}_{clim})}^2}{\sqrt{\sum{(y-{y}_{clim})}^2}}}
$$

- $$\hat{y}$$: forecast (prediction)
- $$y$$: actual dispatch targets (ground truth)
- $${y}_{clim}$$: climatology (mean training set cube)


## 1.4. SSIM

Structural similarity index measure

$$
\text{SSIM}(x, y) = 
\frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}
     {(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

- Higher better
- With $$x=\text{prediction cube}, y=\text{ground truth cube}$$

- A `(11, 11, 11)` gaussian kernel will be convolved (depthwise/separable) with $$x$$ and $$y$$

<img src="resources/week_19/ssim_calculation.png">

- Take the mean across all coordinates

- Three aspects $$SSIM = l(x, y) \times c(x, y) \times s(x, y)$$
    - luminance: $$l(x, y)=2\mu_A \mu_B + C_1 \ /\  2\mu^2_A + \mu^2_B + C_1$$
    - contrast $$c(x, y)=2\sigma_A \sigma_B + C_2 \ /\  2\sigma^2_A + \sigma^2_B + C_2$$
    - structure $$s(x, y)=2\sigma_{AB} + C_3 \ /\  \sigma_A \sigma_B + C_3$$, ($$C_3 = C_2 / 2$$)

- Range: $$[-1,1]$$

- Multiscale variant: *MS-SSIM*
    - does the SSIM in 5 scales, downsampled (2) by average pooling
    - combines it by weighting `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`
    - simplified expression gives: $$\text{MS-SSIM}(x, y) = l_M(x, y)^{\alpha_M} \prod_{j=1}^{M-1} \text{cs}_j(x, y)^{\beta_j}$$
    - With $$l_M(x, y)$$ being the luminance term at the most coarse scale (last) and $$\alpha_M=0.1333$$
    - And $$\text{cs}_j(x, y)=\text{c}_j(x,y)\times\text{s}_j(x,y)$$ at previous scales and $$\beta_j=$$`weights[j]`
    - Problem
        - Our 111 dimension downsampled 4 times will be $$\frac{111}{2^4}\approx6.93 < 11 \text{ (kernel size)}$$
        - `assert smaller_side > (win_size - 1) * (2 ** 4)`
        - ~~Zero-padding if 111 is close to 160~~
        - *Or, smaller kernel: next kernel choice is 7*
        - and $$\sigma \text{of kernel}: 1.5 \rightarrow 1.0$$


## 1.5. LPIPS

<img src="https://camo.githubusercontent.com/a76b7c735cbb76851bab9c7ba7b7cabca3f378b56e504e2d99d09be6f49cb6bd/68747470733a2f2f726963687a68616e672e6769746875622e696f2f5065726365707475616c53696d696c61726974792f696e6465785f66696c65732f666967315f76322e6a7067"/>

<img src="https://richzhang.github.io/PerceptualSimilarity/index_files/network.jpg">

```py
import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

import torch
img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
d = loss_fn_alex(img0, img1)
```
- Will ignore our radial dimension (going with mean of 2D slices)
- Current implementation uses 3 channel images (because of the sot networks alexnet etc.)
- Lower better

## 1.6. PSNR

$$
\text{PSNR}(y, \hat{y}) = 10 \cdot \log_{10} \left( \frac{\text{max}^2(y)}{\text{MSE}(y, \hat{y})} \right)
$$

- Higher better
- Range $$[0,\infty]$$
- Measures image or signal reconstruction quality

## 1.7 Others

- Feature Similarity Index Measure (FSIM)
- Information theoretic-based Statistic Similarity Measure (ISSM)
- Signal to reconstruction error ratio (SRE)
- Spectral angle mapper (SAM)
- Universal image quality index (UIQ)


## 1.8. Which ones I want go with

- RMSE (Root Mean Square)
- NNSE (Normalized NSE)
- ACC (Anomaly Correlation Coefficient)
- MS-SSIM (Multiscale Structural Similarity Index Measure)
- PSNR

<img src = "resources/week_19/noisy_instances.png"/>

| Noise Scale |   RMSE $$\downarrow$$   |   NNSE $$\uparrow$$   |   ACC $$\uparrow$$  |   PSNR $$\uparrow$$   |   MS-SSIM $$\uparrow$$  |
|-------------|----------|----------|---------|-----------|-------------|
| 0.01        | 0.00999  | 0.99481  | 0.99787 | 40.01     | 0.99314     |
| 0.1         | 0.04999  | 0.88430  | 0.94063 | 26.02     | 0.90880     |
| 0.5         | 0.10002  | 0.65633  | 0.81033 | 19.99     | 0.79606     |


# 2. Shrinking high resolution MAS to use as medium resolution

<img src = "resources/week_19/org.gif"/>
<img src = "resources/week_19/shrunk.gif"/>

# 3. Dataset Splitting

- 598 CRs (originally medium) + 54 CRs (shrunk to medium) = 652 CRs = 1069 `vr002.hdf` files from different instruments
- All train/val/test splits will be made on the CRs
- A train/test split once and for all
    - 80% training ~522 CRs
    - 20% testing ~130 CRs
    - Calling the 80% training split D, doing 5-fold cross validation to find hyperparams
    - For each hyperparameter combination:
        - 80% of D for training ~417 CRs
        - 20% of D for validation ~105 CRs

## CV Script done

## Train Script done

## Factorization?

## Metrics?

## Fix flickering?

# Optimal SFNO to beat HUX (vr)

## Hyperparameters

- Factorization:
    - Dense
    - CP
    - Tucker
    - TT


- Modes:
    - 8
    - 16
    - 32
    - 64
    - 128


- Hidden Channels:
    - 64
    - 128
    - 256
    - 512


- Projection/Lifting Ratio:
    - 1
    - 2
    - 4
    - 8
    - 16

## Training/validation strategy

1. 5-fold cross validation to get best hyperparameters
2. Splits are made with carrington rotations, not instruments/datacubes in carrington rotations
3. Train with the best configuration on 80% and report on 20%


## Metrics

1. R^2 score, aka NSE (Nash-Sutcliffe Efficiency)
    - (-inf to 1)
    - Normalize it
    - NNSE = 1 / (2-NSE)
        - (0 to 1)
2. nRMSE
    - Normalized (by the spread of the data) Root MSE
3. Skill score (for comparison with HUX)


<img src="resources/week_19/metrics.png"/>


| Metric                        | Formula                                                                                   | Meaning                                |
|:-------------------------------|:------------------------------------------------------------------------------------------|:---------------------------------------|
| $R^2_{\text{SFNO}}$         | $1 - \frac{\sum (MAS - SFNO)^2}{\sum (MAS - \overline{MAS})^2}$                      | How well SFNO predicts MAS             |
| $\text{NSE}_{\text{SFNO}}$  | $1 - \frac{\sum (MAS - SFNO)^2}{\sum (MAS - \overline{MAS})^2}$                      | Same as $R^2$, common in physics     |
| $\text{Skill}_{\text{SFNO vs HUX}}$ | $1 - \frac{ \sum (MAS - SFNO)^2 }{ \sum (MAS - HUX)^2 }$ | How much SFNO outperforms HUX baseline |

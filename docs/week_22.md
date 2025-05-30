---
layout: default
title: First Paper 2
---

## Summary of this week

Main problem: predictions become worse as we go further.

<img src="resources/week_21/exp_31_1.gif">

1. 3D L2 Loss (exp 34)
2. 2D L2 + L1 Loss (exp 35)
3. Sequential/buffered predicted slices (exp 36)
4. Enlarge MAS and capture more modes (exp 37)
5. Slice-weighted loss function (exp 38)
6. Fine-tuning further channels (exp 39)
7. Train with higher norms for loss (2D L8) (exp 40)
8. Fine tune with higher norms for loss (2D L8) (exp 41)
9. Starting with lower n_modes and increasing it to max possible (not implemented in SFNO, opened an issue)
10. HUX metrics
11. Conclusion & paper outline

## 3D L2 Loss (exp 34)

2D:

$$
\text{L}_2^{(2D)} = \frac{1}{BCR} \sum_{b=1}^{B} \sum_{c=1}^{C} \sum_{r=1}^{R}
\left( \sum_{i=1}^{H} \sum_{j=1}^{W} \left| x_{bcrij} - y_{bcrij} \right|^2 \right)^{1/2}
$$

3D:

$$
\text{L}_2^{(3D)} = \frac{1}{BC} \sum_{b=1}^{B} \sum_{c=1}^{C}
\left( \sum_{r=1}^{R} \sum_{i=1}^{H} \sum_{j=1}^{W} \left| x_{bcrij} - y_{bcrij} \right|^2 \right)^{1/2}
$$

- 8 x 256 architecture
- Trained from scratch on the first 80% CRs
- Reporting results on the last 20%
- 200 epochs

Example 1

3D
<img src="resources/week_22/exp_34_1.gif">
2D
<img src="resources/week_21/exp_31_1.gif">

Example 2

3D
<img src="resources/week_22/exp_34_2.gif">
2D
<img src="resources/week_21/exp_31_2.gif">

Example 3

3D
<img src="resources/week_22/exp_34_3.gif">
2D
<img src="resources/week_21/exp_31_3.gif">

Example 4

3D
<img src="resources/week_22/exp_34_4.gif">
2D
<img src="resources/week_21/exp_31_4.gif">

Example 5

3D
<img src="resources/week_22/exp_34_5.gif">
2D
<img src="resources/week_21/exp_31_5.gif">


Metrics

3D
<img src="resources/week_22/exp_34_metrics.png">

<!-- 2D
<img src="resources/week_21/exp_31_metrics.png"> -->


Numerical comparison

| Loss fn | Val Loss $$\downarrow$$ | RMSE $$\downarrow$$ | NNSE $$\uparrow$$ | MSSSIM $$\uparrow$$ | ACC $$\uparrow$$ | PSNR $$\uparrow$$ |
|--------|----------|------|------|--------|-----|------|
| 2D L2  | 172.1135 | 0.0249 | 0.9927 | 0.9907 | 0.9963 | 39.22 |
| 3D L2  | 1.3854   | 0.0262 | 0.9920 | 0.9869 | 0.9959 | 38.55 |

Numerically, 2D beats 3D. Visually, in my opinion, 3D beats 2D. **Gonna go with 2D.**

## L2 + L1 Loss (exp 35)

- 8 x 256 architecture
- Trained from scratch on the first 80% CRs
- Reporting results on the last 20%
- 200 epochs

Example 1

2D L2 + L1
<img src="resources/week_22/exp_35_1.gif">
2D L2
<img src="resources/week_21/exp_31_1.gif">

Example 2

2D L2 + L1
<img src="resources/week_22/exp_35_2.gif">
2D L2
<img src="resources/week_21/exp_31_2.gif">

Example 3

2D L2 + L1
<img src="resources/week_22/exp_35_3.gif">
2D L2
<img src="resources/week_21/exp_31_3.gif">

Example 4

2D L2 + L1
<img src="resources/week_22/exp_35_4.gif">
2D L2
<img src="resources/week_21/exp_31_4.gif">

Example 5

2D L2 + L1
<img src="resources/week_22/exp_35_5.gif">
2D L2
<img src="resources/week_21/exp_31_5.gif">


Metrics

2D L2 + L1
<img src="resources/week_22/exp_35_metrics.png">

<!-- 2D L2
<img src="resources/week_21/exp_31_metrics.png"> -->


| Loss fn | Val Loss $$\downarrow$$ | RMSE $$\downarrow$$ | NNSE $$\uparrow$$ | MSSSIM $$\uparrow$$ | ACC $$\uparrow$$ | PSNR $$\uparrow$$ |
|-------------|----------|----------|----------|----------|----------|--------|
| 2D L2       | 172.1135 | 0.0249   | 0.9927   | 0.9907   | 0.9963   | 39.22  |
| 2D L2 + L1  | 306.4048 | 0.0264   | 0.9918   | 0.9872   | 0.9959   | 38.41  |

**Gonna go with 2D L2 only.**

## Buffered channels (exp 36)

<img src="resources/week_22/buffered_pred.png">

- 8 x 256 architecture
- Trained from scratch on the first 80% CRs
- Reporting results on the last 20%
- 100 epochs

Example 1

6 * 23 Channels (Buffered)
<img src="resources/week_22/exp_36_1.gif">
138 Channels at once
<img src="resources/week_21/exp_31_1.gif">

Example 2

6 * 23 Channels (Buffered)
<img src="resources/week_22/exp_36_2.gif">
138 Channels at once
<img src="resources/week_21/exp_31_2.gif">

Example 3

6 * 23 Channels (Buffered)
<img src="resources/week_22/exp_36_3.gif">
138 Channels at once
<img src="resources/week_21/exp_31_3.gif">

Example 4

6 * 23 Channels (Buffered)
<img src="resources/week_22/exp_36_4.gif">
138 Channels at once
<img src="resources/week_21/exp_31_4.gif">

Example 5

6 * 23 Channels (Buffered)
<img src="resources/week_22/exp_36_5.gif">
138 Channels at once
<img src="resources/week_21/exp_31_5.gif">

Visually and numerically, buffered is better. But the max absolute error is higher.

Metrics

6 * 23 Channels (Buffered)
<img src="resources/week_22/exp_36_metrics.png">

<!-- 138 Channels at once
<img src="resources/week_21/exp_31_metrics.png"> -->

| Method | RMSE $$\downarrow$$ | NNSE $$\uparrow$$ | MSSSIM $$\uparrow$$ | ACC $$\uparrow$$ | PSNR $$\uparrow$$ |
|----------|----------|----------|----------|----------|--------|
| Buffered | 0.0235   | 0.9933   | 0.9942   | 0.9966   | 39.29  |
| Full Channel | 0.0249   | 0.9927   | 0.9907   | 0.9963   | 39.22  |

## Enlarge the simulations to capture larger modes (exp 37)

- 8 x 64 architecture (8 x 256 didn't fit. i had 8 x 64 results on normal data (exp 33).)
- Trained from scratch on the first 80% CRs
- Reporting results on the last 20%
- 200 epochs
- slice size: `(140, 111, 128)` -> `(140, 222, 256)`
- n_modes: `(110, 64)` (110, 128 in code) -> `(221, 128)` (221, 256 in code)

Example 1

(222, 256)
<img src="resources/week_22/exp_37_1.gif">
(111, 128)
<img src="resources/week_21/exp_33_1.gif">

Example 2

(222, 256)
<img src="resources/week_22/exp_37_2.gif">
(111, 128)
<img src="resources/week_21/exp_33_2.gif">

Example 3

(222, 256)
<img src="resources/week_22/exp_37_3.gif">
(111, 128)
<img src="resources/week_21/exp_33_3.gif">

Example 4

(222, 256)
<img src="resources/week_22/exp_37_4.gif">
(111, 128)
<img src="resources/week_21/exp_33_4.gif">

Example 5

(222, 256)
<img src="resources/week_22/exp_37_5.gif">
(111, 128)
<img src="resources/week_21/exp_33_5.gif">

Metrics

(222, 256)
<img src="resources/week_22/exp_37_metrics.png">

<!-- 138 Channels at once
<img src="resources/week_21/exp_31_metrics.png"> -->


| Method | RMSE $$\downarrow$$ | NNSE $$\uparrow$$ | MSSSIM $$\uparrow$$ | ACC $$\uparrow$$ | PSNR $$\uparrow$$ |
|--------|----------|----------|----------|----------|--------|
| (111, 128) | 0.0292   | 0.9900   | 0.9798   | 0.9949   | 35.77  |
| (222, 256) | 0.0299   | 0.9893   | 0.9748   | 0.9946   | 35.36  |

It is worse.

## Slice weighted loss, further=higher (exp 38)

- 8 x 256 architecture
- Trained from scratch on the first 80% CRs
- Reporting results on the last 20%
- 200 epochs

Example 1

Slice weighted L2
<img src="resources/week_22/exp_38_1.gif">
2D L2
<img src="resources/week_21/exp_31_1.gif">

Example 2

Slice weighted L2
<img src="resources/week_22/exp_38_2.gif">
2D L2
<img src="resources/week_21/exp_31_2.gif">

Example 3

Slice weighted L2
<img src="resources/week_22/exp_38_3.gif">
2D L2
<img src="resources/week_21/exp_31_3.gif">

Example 4

Slice weighted L2
<img src="resources/week_22/exp_38_4.gif">
2D L2
<img src="resources/week_21/exp_31_4.gif">

Example 5

Slice weighted L2
<img src="resources/week_22/exp_38_5.gif">
2D L2
<img src="resources/week_21/exp_31_5.gif">


Metrics

Slice weighted L2
<img src="resources/week_22/exp_38_metrics.png">

<!-- 2D L2
<img src="resources/week_21/exp_31_metrics.png"> -->


| Method | RMSE $$\downarrow$$ | NNSE $$\uparrow$$ | MSSSIM $$\uparrow$$ | ACC $$\uparrow$$ | PSNR $$\uparrow$$ |
|--------|----------|----------|----------|----------|--------|
| L2   | 0.0249   | 0.9927   | 0.9907   | 0.9963   | 39.22  |
| Slice weighted L2 | 0.0250   | 0.9926   | 0.9889   | 0.9963   | 39.12  |

Not that different.

## Fine-tuning slice 24 to 46 (exp 39)

- 8 x 256 architecture
- Fine-tuned on the first 80% CRs
- Reporting results on the last 20%
- 50 epochs

Example 1

Fine-tuned
<img src="resources/week_22/exp_39_1.gif">
Pre-trained
<img src="resources/week_21/exp_31_1.gif">

Example 2

Fine-tuned
<img src="resources/week_22/exp_39_2.gif">
Pre-trained
<img src="resources/week_21/exp_31_2.gif">

Example 3

Fine-tuned
<img src="resources/week_22/exp_39_3.gif">
Pre-trained
<img src="resources/week_21/exp_31_3.gif">

Example 4

Fine-tuned
<img src="resources/week_22/exp_39_4.gif">
Pre-trained
<img src="resources/week_21/exp_31_4.gif">

Example 5

Fine-tuned
<img src="resources/week_22/exp_39_5.gif">
Pre-trained
<img src="resources/week_21/exp_31_5.gif">


Metrics

Fine-tuned
<img src="resources/week_22/exp_39_metrics.png">

| Method | RMSE $$\downarrow$$ | NNSE $$\uparrow$$ | MSSSIM $$\uparrow$$ | ACC $$\uparrow$$ | PSNR $$\uparrow$$ |
|--------|----------|----------|----------|----------|--------|
| Pre-trained   | 0.0249   | 0.9927   | 0.9907   | 0.9963   | 39.22  |
| Fine-tuned | 0.1443   | 0.7749   | 0.7882   | 0.8294   | 20.42  |

Per-slice fine-tuning is wrong. 90% of weights are spectral not spatial.

## Training with higher norms for loss (2D L8) (exp 40)

$$
\text{L}_8^{(2D)} = \frac{1}{BCR} \sum_{b=1}^{B} \sum_{c=1}^{C} \sum_{r=1}^{R}
\left( \sum_{i=1}^{H} \sum_{j=1}^{W} \left| x_{bcrij} - y_{bcrij} \right|^8 \right)^{1/8}
$$

- 8 x 256 architecture
- Trained from scratch on the first 80% CRs
- Reporting results on the last 20%
- 200 epochs

Example 1

L8
<img src="resources/week_22/exp_40_1.gif">
L2
<img src="resources/week_21/exp_31_1.gif">

Example 2

L8
<img src="resources/week_22/exp_40_2.gif">
L2
<img src="resources/week_21/exp_31_2.gif">

Example 3

L8
<img src="resources/week_22/exp_40_3.gif">
L2
<img src="resources/week_21/exp_31_3.gif">

Example 4

L8
<img src="resources/week_22/exp_40_4.gif">
L2
<img src="resources/week_21/exp_31_4.gif">

Example 5

L8
<img src="resources/week_22/exp_40_5.gif">
L2
<img src="resources/week_21/exp_31_5.gif">


Metrics

L8
<img src="resources/week_22/exp_40_metrics.png">

| Loss | RMSE $$\downarrow$$ | NNSE $$\uparrow$$ | MSSSIM $$\uparrow$$ | ACC $$\uparrow$$ | PSNR $$\uparrow$$ |
|--------|----------|----------|----------|----------|--------|
| L8 | ***0.0233***   | ***0.9934***   | 0.9877   | ***0.9967***   | 35.83  |
| L2 | 0.0249   | 0.9927   | ***0.9907***   | 0.9963   | ***39.22***  |

L8 is better but has more flickering/ringing...


## Fine-tuning L2-trained on higher norms for loss (2D L8) (exp 41)

- 8 x 256 architecture
- Fine-tuned on the first 80% CRs
- Reporting results on the last 20%
- 50 epochs

Example 1

L8 Fine-tuned
<img src="resources/week_22/exp_41_1.gif">
L2
<img src="resources/week_21/exp_31_1.gif">

Example 2

L8 Fine-tuned
<img src="resources/week_22/exp_41_2.gif">
L2
<img src="resources/week_21/exp_31_2.gif">

Example 3

L8 Fine-tuned
<img src="resources/week_22/exp_41_3.gif">
L2
<img src="resources/week_21/exp_31_3.gif">

Example 4

L8 Fine-tuned
<img src="resources/week_22/exp_41_4.gif">
L2
<img src="resources/week_21/exp_31_4.gif">

Example 5

L8 Fine-tuned
<img src="resources/week_22/exp_41_5.gif">
L2
<img src="resources/week_21/exp_31_5.gif">


L8 Fine-tuned
<img src="resources/week_22/exp_41_metrics.png">

| Loss | RMSE $$\downarrow$$ | NNSE $$\uparrow$$ | MSSSIM $$\uparrow$$ | ACC $$\uparrow$$ | PSNR $$\uparrow$$ |
|--------|----------|----------|----------|----------|--------|
| L2 | 0.0235   | 0.9933   | 0.9942   | 0.9966   | 39.29  |
| L8 Fine-tuned | 0.0281   | 0.9907   | 0.9809   | 0.9953   | 33.73  |


## HUX metrics

| Method | RMSE $$\downarrow$$ | NNSE $$\uparrow$$ | MSSSIM $$\uparrow$$ | ACC $$\uparrow$$ | PSNR $$\uparrow$$ |
|--------|----------|----------|----------|----------|--------|
| Buffered L2 2D | 0.0235 (code units)  | 0.9933   | 0.9942   | 0.9966   | 39.29  |
| Full Channel L2 2D | 0.0249 (code units)  | 0.9927   | 0.9907   | 0.9963   | 39.22  |
| HUX    | 40.7198 (km/s) | 0.9149   | 0.9723   | 0.9584   | 27.82  |


## Conclusion and paper outline

Here's what I will say:

<i>We found out that 8 x 256 performs best. But in latter slices predictions became worse. Thus we took the buffered channels approach and its our optimal model. **One problem though, last slice?**</i>

- Introduction
    - What is the solar wind
        - $$J, B, V, \rho, ...$$
    - Importance of surrogates for this
    - Numerical solvers
        - MAS
        - Others?
    - Data-driven/ML solvers
        - Why they are important
        - Any similar work
    - Our problem
        - why we started with velocity
        - why only radial
- Background
    - Numerical solvers
        - MAS
        - Others?
    - Data-driven/ML solvers
        - Any similar work
        - Traditional DL (i.e., in spatial dims)
        - Operator learning
            - DeepONet
            - FNO
            - SFNO
- Methods
    - Dataset
        - Why velocity
        - Why only radial?
    - Baseline: HUX
    - SFNO
    - Training strategy
        - Dataset split
        - Cross val
        - Hyperparams
        - Loss
        - Evaluation metrics
- Results and Discussion


***ICDM deadline is 8 AM next Saturday.***
***CogMI next, with 21 June deadline.***
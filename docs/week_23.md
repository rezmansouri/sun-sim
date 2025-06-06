---
layout: default
title: First Paper 3
---

## Important Regions (Edge detection)

<img src="resources/week_23/edge_1.gif"/>
<img src="resources/week_23/edge_2.gif"/>
<img src="resources/week_23/edge_3.gif"/>

## Sequential training (last experiments for paper 1)

<img src="resources/week_23/buffered.png"/>


## 20 channels (exp 42)

7 * 20 Channels
<img src="resources/week_23/exp_42_1.gif">
139 Channels at once
<img src="resources/week_21/exp_31_1.gif">

7 * 20 Channels
<img src="resources/week_23/exp_42_2.gif">
139 Channels at once
<img src="resources/week_21/exp_31_2.gif">

7 * 20 Channels
<img src="resources/week_23/exp_42_3.gif">
139 Channels at once
<img src="resources/week_21/exp_31_3.gif">

7 * 20 Channels
<img src="resources/week_23/exp_42_4.gif">
139 Channels at once
<img src="resources/week_21/exp_31_4.gif">

7 * 20 Channels
<img src="resources/week_23/exp_42_5.gif">
139 Channels at once
<img src="resources/week_21/exp_31_5.gif">

Metrics (everywhere)

| Method | NNSE $$\uparrow$$ | MSSSIM $$\uparrow$$ | ACC $$\uparrow$$ | PSNR $$\uparrow$$ |
|--------|----------|----------|----------|----------|
| 20 Channels | 0.9939   | 0.9944   | 0.9969   | 40.27  |
| 139 Channels at once | 0.9933   | 0.9942   | 0.9966   | 39.29  |


<img src="resources/week_23/exp_42_metrics.png">


## 10 channels (exp 43)

14 * 10 Channels
<img src="resources/week_23/exp_43_1.gif">
139 Channels at once
<img src="resources/week_21/exp_31_1.gif">

14 * 10 Channels
<img src="resources/week_23/exp_43_2.gif">
139 Channels at once
<img src="resources/week_21/exp_31_2.gif">

14 * 10 Channels
<img src="resources/week_23/exp_43_3.gif">
139 Channels at once
<img src="resources/week_21/exp_31_3.gif">

14 * 10 Channels
<img src="resources/week_23/exp_43_4.gif">
139 Channels at once
<img src="resources/week_21/exp_31_4.gif">

14 * 10 Channels
<img src="resources/week_23/exp_43_5.gif">
139 Channels at once
<img src="resources/week_21/exp_31_5.gif">


Metrics (everywhere)

| Method | NNSE $$\uparrow$$ | MSSSIM $$\uparrow$$ | ACC $$\uparrow$$ | PSNR $$\uparrow$$ |
|--------|----------|----------|----------|----------|
| 10 Channels  | 0.9941   | 0.9945   | 0.9970   | 40.29  |
| 20 Channels  | 0.9939   | 0.9944   | 0.9969   | 40.27  |
| 139 Channels at once   | 0.9933   | 0.9942   | 0.9966   | 39.29  |
| HUX    | 0.9149   | 0.9723   | 0.9584   | 27.82  |


<img src="resources/week_23/exp_43_metrics.png">

Epoch 10 we had a jump. This is the same plots with that removed:

<img src="resources/week_23/exp_43_metrics_2.png">


Some more instances:

<img src="resources/week_23/exp_43_6.gif">

<img src="resources/week_23/exp_43_7.gif">


MSE:

$$
\text{MSE} = \frac{1}{RHW} \sum_{r=1}^{R}
\sum_{i=1}^{H} \sum_{j=1}^{W} \left| x_{bcrij} - y_{bcrij} \right|^2
$$

<img src="resources/week_23/mse.png">
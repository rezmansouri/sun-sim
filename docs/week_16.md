# SFNO on PFSS, MHD rho/vr and DeepONet

## Experiment 14

- PFSS
- `br`
- Model: SFNO

```
n_modes=64
hidden_channels=128
epochs=500
train_size=35
validation_size=12
```

<img src="resources/week_16/exp_14_1.gif">
<img src="resources/week_16/exp_14_2.gif">
<img src="resources/week_16/exp_14_3.gif">

## Experiment 15

- MHD
- `vr` and `rho` interleaved
- Model: SFNO

```
n_modes=64
hidden_channels=128
epochs=500
train_size=680
validation_size=227
```

<img src="resources/week_16/exp_15_1_rho.gif">
<img src="resources/week_16/exp_15_1_v.gif">

<img src="resources/week_16/exp_15_2_rho.gif">
<img src="resources/week_16/exp_15_2_v.gif">

## Experiment 16

- MHD
- `vr` and `rho` interleaved
- Model: CartesianProductDeepONet

<img src="https://www.mdpi.com/algorithms/algorithms-15-00325/article_deploy/html/images/algorithms-15-00325-g001.png">

```
epochs=500
train_size=680
validation_size=227
```

<img src="resources/week_16/exp_16_1.gif">
<img src="resources/week_16/exp_16_2.gif">
<img src="resources/week_16/exp_17_loss.png">



## References
- A blog post on a DeepONet simple example: https://towardsdatascience.com/operator-learning-via-physics-informed-deeponet-lets-implement-it-from-scratch-6659f3179887/
---
layout: default
title: failed attention enhanced multimodal sfno
---

## Multimodal SFNO

<img src="resources/week_37/4.jpg"/>

### Exp 50

$$V_r$$ and $$J_r$$

All the data we had is used for train/test 80/20.

No autoregression, all shells at once.

```py
{
  "num_epochs": 100,
  "batch_size": 32,
  "hidden_channels": 64,
  "encoder_hidden_channels": 64,
  "n_layers": 4,
  "loss_fn": "l2"
}
```

#### CR2284

<img src="resources/week_39/exp50-cr2284.gif"/>


## Multimodal SFNO + Attention

### Exp 52

<img src="resources/week_43/model.jpg"/>

```py
{
  "num_epochs": 100,
  "batch_size": 32,
  "hidden_channels": 64,
  "encoder_hidden_channels": 64,
  "n_layers": 4,
  "attention_heads": 4,
  "attention_hidden_dim": 64,
  "loss_fn": "l2"
}
```
#### CR2284

<img src="resources/week_43/exp52-cr2284.gif"/>
<img src="resources/week_43/dist.jpg"/>
<img src="resources/week_43/exp50-vs-exp52-vr-slice-139.png"/>
<img src="resources/week_43/exp50-vs-exp52-jr-slice-10.png"/>


## Misc. and next steps

1. Physical loss
  - For journal paper
  - Current situation: <a href="https://rezmansouri.github.io/sun-sim/week_41.html" target="_blank">equation 1</a> seemed to be working according to pete in last meeting
  - Train multimodal SFNO with and without the physics loss
  - No baseline? Show the effectiveness of physics loss presence
  - Eq. 6 is better application-wise, working on it

2. DeepONet is deprecated?, vignesh might have to give up on it

3. <a href="https://arxiv.org/abs/2510.05620" target="_blank">Monte-carlo FNO from ICMLA '25</a>
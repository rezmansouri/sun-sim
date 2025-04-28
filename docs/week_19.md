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
    -128


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
2. Train with the best configuration on 80% and report on 20%


## Metrics

1. R^2 score, aka NSE (Nash-Sutcliffe Efficiency)
2. nRMSE
3. Skill score (for comparison with HUX)


<img src="resources/week_19/metrics.png"/>


<!-- | Metric                        | Formula                                                                                   | Meaning                                |
|:-------------------------------|:------------------------------------------------------------------------------------------|:---------------------------------------|
| $R^2_{\text{SFNO}}$         | $1 - \frac{\sum (MAS - SFNO)^2}{\sum (MAS - \overline{MAS})^2}$                      | How well SFNO predicts MAS             |
| $\text{NSE}_{\text{SFNO}}$  | $1 - \frac{\sum (MAS - SFNO)^2}{\sum (MAS - \overline{MAS})^2}$                      | Same as $R^2$, common in physics     |
| $\text{Skill}_{\text{SFNO vs HUX}}$ | $1 - \frac{ \sum (MAS - SFNO)^2 }{ \sum (MAS - HUX)^2 }$ | How much SFNO outperforms HUX baseline | -->

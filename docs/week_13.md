## Pointnet and DeepSphere

### Exp 9:

- Slice[0] to slice[i] for all cubes
- Features: `(phi, theta, r[target_slice], intensity[bc])`
- Output: `intensity[target_slice]`

#### Losses

<img src="resources/week_13/exp9_loss.png">
<img src="resources/week_13/exp9_logloss.png">

#### Comparison
<img src="resources/week_13/exp9_result_1.gif">
<img src="resources/week_13/exp9_result_2.gif">
<img src="resources/week_13/exp9_result_3.gif">

### Exp 10:

- Slice[0] to slice[i] for all cubes
- Features: `(x[target_slice], y[target_slice], z[target_slice], intensity[bc])`
- Output: `intensity[target_slice]`

#### Losses

<img src="resources/week_13/exp10_loss.png">
<img src="resources/week_13/exp10_logloss.png">

#### Comparison
<img src="resources/week_13/exp10_result_1.gif">
<img src="resources/week_13/exp10_result_2.gif">
<img src="resources/week_13/exp10_result_3.gif">

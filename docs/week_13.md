## Pointnet and Google's SphericalCNN

### Exp 9:

- Slice[0] to slice[i] for all cubes
- Features: `(x[target_slice], y[target_slice], z[target_slice], intensity[bc])`
- Output: `intensity[target_slice]`

#### Losses

<img src="resources/week_13/exp9_loss.png">
<img src="resources/week_13/exp9_logloss.png">

#### Comparison

- Validation
<img src="resources/week_13/exp9_result_test.gif">
<img src="resources/week_13/exp9_result_test2.gif">


- Training
<img src="resources/week_13/exp9_result_train.gif">

### Exp 10:

- Slice[0] to slice[i] for all cubes
- Features: `(phi, theta, r[target_slice], intensity[bc])`
- Output: `intensity[target_slice]`

#### Losses

<img src="resources/week_13/exp10_loss.png">
<img src="resources/week_13/exp10_logloss.png">

#### Comparison

- Validation
<img src="resources/week_13/exp10_result_test.gif">
<img src="resources/week_13/exp10_result_test2.gif">


- Training
<img src="resources/week_13/exp10_result_train.gif">


### DeepSphere for weather

Uses 
- SphericalUNet
- Transforms the data into HEALPix Grid
<img src="https://healpix.jpl.nasa.gov/images/healpixGridRefinement.jpg">
<img src="resources/week_13/healpix.png">


### Spherical-CNN from Google

- Written in Jax, Flax and Linen instead of Pytorch and TensorFlow
- Allows turning off spin equivariance
- I have written an auto-encoder so far for bc-to-slice prediction

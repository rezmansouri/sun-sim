<style>
        .image-row {
            display: flex;
            justify-content: space-around; /* Distribute space around the images */
            margin: 20px 0;
        }
        .image-row img {
            width: 100%; /* Adjust the width as needed */
            height: auto;
            margin: 0px; /* Optional: space between images */
        }
    </style>

<h1>Week two/three: 23 July - 7 August</h1>

## 1. Navier-Stokes with PINNs

### 1.1. Equations
Incompressible fluid flow in a cylinder-like 2D environment x, y, t -> p, (u, v)

<img src="resources/week_3/new_f.svg">

<img src="resources/week_3/new_g.svg">

<img src="resources/week_3/new_assm.svg">

So our PINN will output: <img src="resources/week_3/new_psi and p.svg">

And we'll try to minimize:

<img src="resources/week_3/new_loss.svg">

### 1.2. Data
#### 1.2.1 Specs
- Coordinates (x, y), #: 5000 <span style="color:blue">Input</span>
- Time (t), #: 200, range: [0, 0.1, 0.2, ..., 20] <span style="color: blue">Input</span>
- Pressure (p), #: 5000 x 200 <span style="color: green">Output</span>
- Velocity (u, v), #: 5000 x 200 <span style="color: green">Output</span>

### 1.2.2. Train / test split
5000 random instances from 5000 x 200 instances for training.

Using 100 consecutive instances from the whole 5000 x 200 for plotting comparisons.


### 1.3. Methods

#### 1.3.1. Architecture
MLP
- 3 input nodes
- 8 x 20 hidden layers with tanh
- 2 output nodes
- lambda 1 and lambda 2 (initialized with 0)

<img src="resources/week_3/new_arch.png">

#### 1.3.2. Optimizer

<span style="color:red"><s>SGD, Momentum, RMSProp, Adam, ...</s></span>

<span style="color:green">L-BFGS<span>:

- Is deterministic, unlike stochastic ones
- More suitable for physics smooth loss landscapes
- Faster convergence
- Memory efficient
- For solving PDEs, is more precise

### 1.4. Results

#### 1.4.1. Learning curve

<img src="resources/week_3/new_curve.svg">

#### 1.4.2. Pressure t: [1, 20]

<div class="image-row">
        <img src="resources/week_3/new_pressure_gt.gif" alt="Image 1">
        <img src="resources/week_3/new_pressure_pred.gif" alt="Image 2">
</div>

#### 1.4.3. Velocity (u) t: [1, 20]

<div class="image-row">
        <img src="resources/week_3/new_u_gt.gif" alt="Image 1">
        <img src="resources/week_3/new_u_pred.gif" alt="Image 2">
</div>


#### 1.4.4. Velocity (v) t: [1, 20]

<div class="image-row">
        <img src="resources/week_3/new_v_gt.gif" alt="Image 1">
        <img src="resources/week_3/new_v_pred.gif" alt="Image 2">
</div>

#### 1.4.5. PDE Solutions

<table>
  <tr>
    <th></th>
    <th>f</th>
    <th>g</th>
  </tr>
  <tr>
    <td>Correct PDE</td>
    <td><img src="resources/week_3/new_org_f.svg" alt="Correct PDE f"></td>
    <td><img src="resources/week_3/new_org_g.svg" alt="Correct PDE g"></td>
  </tr>
  <tr>
    <td>Identified PDE (clean data)</td>
    <td><img src="resources/week_3/new_clean_f.svg" alt="Identified PDE (clean data) f"></td>
    <td><img src="resources/week_3/new_clean_g.svg" alt="Identified PDE (clean data) g"></td>
  </tr>
  <tr>
    <td>Identified PDE (1% noise)</td>
    <td><img src="resources/week_3/new_noisy_f.svg" alt="Identified PDE (1% noise) f"></td>
    <td><img src="resources/week_3/new_noisy_g.svg" alt="Identified PDE (1% noise) g"></td>
  </tr>
</table>


#### References
- <a href="https://maziarraissi.github.io/PINNs/">Maziar Raissi's GitHub (TensorFlow)</a>
- <a href="https://github.com/rezaakb/pinns-torch/tree/main">Reza Akbari PINNsTorch Package</a>
- <a href="https://github.com/ComputationalDomain/PINNs/tree/main/Cylinder-Wake">ComputationDomain Navier-Stokes in Torch</a>
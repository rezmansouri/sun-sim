<style>
        .image-row {
            display: flex;
            justify-content: space-around; /* Distribute space around the images */
            margin: 20px 0;
        }
        .image-row img {
            width: 70%; /* Adjust the width as needed */
            height: auto;
            margin: 0px; /* Optional: space between images */
        }
    </style>

<h1>Week two/three: 23 July - 7 August</h1>

## 1. Navier-Stokes with PINNs

### 1.1. Equations
Incompressible fluid flow in a cylinder-like 2D environment x, y, t -> p, (u, v)

<img src="resources/week_3/f.svg">

<img src="resources/week_3/g.svg">

<img src="resources/week_3/assm.svg">

So our PINN will output: <img src="resources/week_3/psi and p.svg">

And we'll try to minimize:

<img src="resources/week_3/loss.svg">

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

<img src="resources/week_3/nn.png">

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

<img src="resources/week_3/curve.svg">

#### 1.4.2. Pressure t: [1, 20]

<div class="image-row">
        <img src="resources/week_3/pressure_gt.gif" alt="Image 1">
        <img src="resources/week_3/pressure_pred.gif" alt="Image 2">
</div>

#### 1.4.3. Velocity (u) t: [1, 20]

<div class="image-row">
        <img src="resources/week_3/u_gt.gif" alt="Image 1">
        <img src="resources/week_3/u_pred.gif" alt="Image 2">
</div>


#### 1.4.4. Velocity (v) t: [1, 20]

<div class="image-row">
        <img src="resources/week_3/v_gt.gif" alt="Image 1">
        <img src="resources/week_3/v_pred.gif" alt="Image 2">
</div>
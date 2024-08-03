<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<h1>Week two/three: 23 July - 7 August</h1>

## 1. Navier-Stokes with PINNs

### 1.1. Equations
Incompressible fluid flow in a cylinder-like environment

x, y, t -> p, (u, v)

<img src="resources/week_3/f.svg">

<img src="resources/week_3/g.svg">


### 1.2. Data
#### 1.2.1 Specs
- Coordinates (x, y), #: 5000 <span style="color:blue">Input</span>
- Time (t), #: 200, range: [0, 0.1, 0.2, ..., 20] <span style="color: blue">Input</span>
- Pressure (p), #: 5000 x 200 <span style="color: lightgreen">Output</span>
- Velocity (u, v), #: 5000 x 200 <span style="color: lightgreen">Output</span>

### 1.2.2. Train / test split
5000 random instances from 5000 x 200 instances for training.
Using the whole 5000 x 200 for plotting comparisons.


### 1.3. Methods

#### 1.3.1. Architecture
MLP
- 3 input nodes
- 8 x 20 hidden layers with tanh
- 2 output nodes

<img src="resources/week_3/nn.svg">

#### 1.3.2. Optimizer

<span style="color:red"><s>SGD, Momentum, RMSProp, Adam, ...</s></span>

<span style="color:lightgreen">L-BFGS<span>:

- Is deterministic, unlike stochastic ones
- More suitable for physics smooth loss landscapes
- Faster convergence
- Memory efficient
- For solving PDEs, is more precise

### 1.4. Results

<img src="resources/week_3/curve.svg">

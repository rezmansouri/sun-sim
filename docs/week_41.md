---
layout: default
title: eq. 6 and eq. 1
---

## Update

Use the smaller common grid between the components for interpolation: (140, 110, 128)

## Equation 1 in static state (r, theta and phi):

$$
\nabla \times B = \frac{4\pi}{c}(\text{or }\mu_0)J
$$



$$
(\nabla \times \mathbf{B})_r = \frac{1}{r \sin\theta}
\left[
\frac{\partial}{\partial \theta}(B_\phi \sin\theta)
- \frac{\partial B_\theta}{\partial \phi}
\right]
$$

$$
(\nabla \times \mathbf{B})_\theta = \frac{1}{r}
\left[
\frac{1}{\sin\theta} \frac{\partial B_r}{\partial \phi}
- \frac{\partial}{\partial r}(r B_\phi)
\right]
$$

$$
(\nabla \times \mathbf{B})_\phi = \frac{1}{r}
\left[
\frac{\partial}{\partial r}(r B_\theta)
- \frac{\partial B_r}{\partial \theta}
\right]
$$


Error is

$$
\sqrt{\text{residual}_r^2 + \text{residual}_{\theta}^2 + \text{residual}_{\phi}^2}
$$

## cgs system

$$
\nabla \times B = \frac{4\pi}{c}J
$$

CR1833

$$
\langle |\nabla \times B|_r \rangle = 1.942 \times 10^{-19}, \quad
$$
$$
\langle |\nabla \times B|_\theta \rangle = 2.487 \times 10^{-19}, \quad
$$
$$
\langle |\nabla \times B|_\phi \rangle = 3.839 \times 10^{-19}
$$

$$
\langle |\mu_0 J|_r \rangle = 1.934 \times 10^{-19}, \quad
$$
$$
\langle |\mu_0 J|_\theta \rangle = 2.601 \times 10^{-19}, \quad
$$
$$
\langle |\mu_0 J|_\phi \rangle = 3.856 \times 10^{-19}
$$

$$
\langle |R|_r \rangle = 4.579 \times 10^{-21}, \quad
$$
$$
\langle |R|_\theta \rangle = 6.601 \times 10^{-20}, \quad
$$
$$
\langle |R|_\phi \rangle = 5.804 \times 10^{-21}, \quad
$$
$$
\langle |R| \rangle = 6.895 \times 10^{-20}
$$

$$
\frac{\langle |\nabla \times B|_r \rangle}{\langle |\mu_0 J|_r \rangle} = 1.004, \quad
\frac{\langle |\nabla \times B|_\theta \rangle}{\langle |\mu_0 J|_\theta \rangle} = 0.956, \quad
\frac{\langle |\nabla \times B|_\phi \rangle}{\langle |\mu_0 J|_\phi \rangle} = 0.996
$$

$$
\text{Residual ratio: } 
\frac{\| \nabla \times B - \mu_0 J \|}{\| \nabla \times B \| + \| \mu_0 J \|} = 4.139 \times 10^{-2}
$$

$$
\|\nabla \times B\|_\text{total} = 8.268 \times 10^{-19}, \quad\\
\|\mu_0 J\|_\text{total} = 8.390 \times 10^{-19}, \quad\\
\|R\|_\text{total} = 6.895 \times 10^{-20}
$$

$$
\text{Relative residual distribution: } 
$$
$$
\text{mean(rel-res)} = 1.228 \times 10^{-1}, \quad
$$
$$
\text{median(rel-res)} = 9.532 \times 10^{-2}, \quad
$$
$$
p_{90}(\text{rel-res}) = 2.280 \times 10^{-1}, \quad
$$
$$
p_{99}(\text{rel-res}) = 6.218 \times 10^{-1}
$$



## mks system

$$
\nabla \times B = \mu_0J
$$

$$
\mu_0: \text{permeability of free space (vacuum)} = 4 \pi \times 10^{-7} \text{Henry/m or T.m/A}
$$

CR1833


$$
\langle |\nabla \times B|_r \rangle = 1.942 \times 10^{-19}, \quad
$$
$$
\langle |\nabla \times B|_\theta \rangle = 2.487 \times 10^{-19}, \quad
$$
$$
\langle |\nabla \times B|_\phi \rangle = 3.839 \times 10^{-19}
$$

$$
\langle |\mu_0 J|_r \rangle = 1.934 \times 10^{-19}, \quad
$$
$$
\langle |\mu_0 J|_\theta \rangle = 2.601 \times 10^{-19}, \quad
$$
$$
\langle |\mu_0 J|_\phi \rangle = 3.856 \times 10^{-19}
$$

$$
\langle |R|_r \rangle = 4.579 \times 10^{-21}, \quad
$$
$$
\langle |R|_\theta \rangle = 6.601 \times 10^{-20}, \quad
$$
$$
\langle |R|_\phi \rangle = 5.804 \times 10^{-21}, \quad
$$
$$
\langle |R| \rangle = 6.895 \times 10^{-20}
$$

$$
\frac{\langle |\nabla \times B|_r \rangle}{\langle |\mu_0 J|_r \rangle} = 1.004, \quad
\frac{\langle |\nabla \times B|_\theta \rangle}{\langle |\mu_0 J|_\theta \rangle} = 0.956, \quad
\frac{\langle |\nabla \times B|_\phi \rangle}{\langle |\mu_0 J|_\phi \rangle} = 0.996
$$

$$
\text{Residual ratio: } 
\frac{\| \nabla \times B - \mu_0 J \|}{\| \nabla \times B \| + \| \mu_0 J \|} = 4.139 \times 10^{-2}
$$

$$
\|\nabla \times B\|_\text{total} = 8.268 \times 10^{-19}, \quad\\
\|\mu_0 J\|_\text{total} = 8.390 \times 10^{-19}, \quad\\
\|R\|_\text{total} = 6.895 \times 10^{-20}
$$

$$
\text{Relative residual distribution: } 
$$
$$
\text{mean(rel-res)} = 1.228 \times 10^{-1}, \quad
$$
$$
\text{median(rel-res)} = 9.532 \times 10^{-2}, \quad
$$
$$
p_{90}(\text{rel-res}) = 2.280 \times 10^{-1}, \quad
$$
$$
p_{99}(\text{rel-res}) = 6.218 \times 10^{-1}
$$



## Equation 6 in static state:

$$
\rho\left(-\Omega_{\text{rot}}\frac{\partial{v_r}}{\partial \phi}+{v} \cdot \nabla {v}\right)=\frac{1}{c} {J} \times {B}-\nabla P+\rho {g}+\nabla \cdot(\nu \rho \nabla {v})
$$

### Update:

$$
(J \times B)_r = J_{\theta}B_{\phi} - J_{\phi}B_{\theta}
$$

## cgs system

$$
r = r \times 6.96 \times 10^{10}\ \text{cm}
$$

$$
v_r = v_r \times 481.3711 \times 10^{5}\ \text{cm/s}
$$

$$
\rho = \rho \times 1.6726 \times 10^{-16}\ \text{g/cm}^3
$$

$$
p = p \times 0.3875717\ \text{dyn/cm}^2
$$

$$
j_r = j_r \times 0.07558\ \text{statamp/cm}^2
$$

$$
B_r = B_r \times 2.2068908\ \text{G}
$$

$$
G = 6.67430 \times 10^{-8}\ \text{cm}^3\,\text{g}^{-1}\,\text{s}^{-2}
$$

$$
M_{\odot} = 1.9885 \times 10^{33}\ \text{g}
$$

$$
\Omega_{\text{rot}} = 2.84 \times 10^{-6}\ \text{rad/s}
$$

$$
c = 2.99792458 \times 10^{10}\ \text{cm/s}
$$

Update:

$$
\nu = 0.005 \times 3.350342628857710 \times {10}^{18}\ \text{cm}^2/\text{s}
$$


CR1833

Error is \|RHS - LHS\|.

$$
\begin{aligned}
&\rho\left(-\Omega \frac{\partial v_r}{\partial \phi}\right) &: 3.324 \times 10^{-21} \\
&\rho v_r \frac{\partial v_r}{\partial r} &: 7.085 \times 10^{-21} \\
&\frac{(J\times B)_r}{c} &: 1.737 \times 10^{-22} \\
&-\frac{\partial p}{\partial r} &: 5.712 \times 10^{-21} \\
&\rho g &: 8.437 \times 10^{-22} \\
&\text{Viscous} &: 3.172 \times 10^{-24} \\
&\text{Residual mean} &: 1.813 \times 10^{-21} \\
&\text{Residual ratio} &: 1.057 \times 10^{-1} \\
\end{aligned}
$$

$$
\begin{aligned}
&\text{Relative residual distribution:} \\
&\text{mean(rel-res)} &: 2.157 \times 10^{-1} \\
&\text{median(rel-res)} &: 1.402 \times 10^{-1} \\
&\text{p90(rel-res)} &: 4.872 \times 10^{-1} \\
&\text{p99(rel-res)} &: 1.061 \times 10^{0} \\
\end{aligned}
$$




## mks system


$$
r = r \times 6.96 \times 10^{8}\ \text{m}
$$

$$
v_r = v_r \times 481.3711 \times 10^{3}\ \text{m/s}
$$

$$
\rho = \rho \times 1.6726 \times 10^{-13}\ \text{kg/m}^3
$$

$$
p = p \times 0.03875717\ \text{Pa}
$$

Verified:

$$
j_r = j_r \times 2.52 \times 10^{-7}\ \text{A/m}^2
$$

$$
B_r = B_r \times 2.2068908 \times 10^{-4}\ \text{T}
$$

$$
G = 6.67430 \times 10^{-11}\ \text{m}^3\,\text{kg}^{-1}\,\text{s}^{-2}
$$

$$
M_{\odot} = 1.9885 \times 10^{30}\ \text{kg}
$$

$$
\Omega_{\text{rot}} = 2.84 \times 10^{-6}\ \text{rad/s}
$$

$$
c = 2.99792458 \times 10^{8}\ \text{m/s}
$$

Update:

$$
\nu = 0.005 \times  10^{-4} \times 3.350342628857710 * {10}^{18}\ \text{m}^2/\text{s}
$$


CR1833

$$
\begin{aligned}
&\rho\left(-\Omega \frac{\partial v_r}{\partial \phi}\right) &: 3.324 \times 10^{-20} \\
&\rho v_r \frac{\partial v_r}{\partial r} &: 7.085 \times 10^{-20} \\
&\frac{(J\times B)_r}{c} &: 5.791 \times 10^{-30} \\
&-\frac{\partial p}{\partial r} &: 5.712 \times 10^{-20} \\
&\rho g &: 8.437 \times 10^{-21} \\
&\text{Viscous} &: 3.172 \times 10^{-23} \\
&\text{Residual mean} &: 1.868 \times 10^{-20} \\
&\text{Residual ratio} &: 1.101 \times 10^{-1}\\
\end{aligned}
$$

$$
\begin{aligned}
&\text{Relative residual distribution:} \\
&\text{mean(rel-res)} &: 2.145 \times 10^{-1} \\
&\text{median(rel-res)} &: 1.390 \times 10^{-1} \\
&\text{p90(rel-res)} &: 4.849 \times 10^{-1} \\
&\text{p99(rel-res)} &: 1.089 \times 10^{0} \\
\end{aligned}
$$

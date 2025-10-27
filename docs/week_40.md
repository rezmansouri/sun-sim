---
layout: default
title: eq. 6 and eq. 1
---

# Updates:

- No fourier approximation for derivatives (in phis)
- All derivatives are computed using <a href="https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf">second-order central differences</a> using torch gradient

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
\langle |\nabla \times B|_r \rangle = 1.919 \times 10^{-17}, \quad\\
\langle |\nabla \times B|_\theta \rangle = 2.621 \times 10^{-17}, \quad\\
\langle |\nabla \times B|_\phi \rangle = 3.975 \times 10^{-17}
$$

$$
\langle |\mu_0 J|_r \rangle = 1.984 \times 10^{-17}, \quad\\
\langle |\mu_0 J|_\theta \rangle = 2.893 \times 10^{-17}, \quad\\
\langle |\mu_0 J|_\phi \rangle = 4.113 \times 10^{-17}\\
$$

$$
\langle |R|_r \rangle = 1.709 \times 10^{-18}, \quad\\
\langle |R|_\theta \rangle = 8.915 \times 10^{-18}, \quad\\
\langle |R|_\phi \rangle = 4.930 \times 10^{-18}, \quad\\
\langle |R| \rangle = 1.241 \times 10^{-17}
$$

$$
\frac{\langle |\nabla \times B|_r \rangle}{\langle |\mu_0 J|_r \rangle} = 0.9671, \quad
\frac{\langle |\nabla \times B|_\theta \rangle}{\langle |\mu_0 J|_\theta \rangle} = 0.9059, \quad
\frac{\langle |\nabla \times B|_\phi \rangle}{\langle |\mu_0 J|_\phi \rangle} = 0.9666
$$

$$
\text{Residual ratio: } \frac{\| \nabla \times B - \mu_0 J \|}{\| \nabla \times B \| + \| \mu_0 J \|} = 7.090 \times 10^{-2}
$$

$$
\|\nabla \times B\|_\text{total} = 8.515 \times 10^{-17}, \quad\\
\|\mu_0 J\|_\text{total} = 8.990 \times 10^{-17}, \quad\\
\|R\|_\text{total} = 1.241 \times 10^{-17}
$$

$$
\text{Relative residual distribution: } \\
\text{mean(rel-res)} = 1.416 \times 10^{-1}, \quad\\
\text{median(rel-res)} = 1.052 \times 10^{-1}, \quad\\
p_{90}(\text{rel-res}) = 2.671 \times 10^{-1}, \quad\\
p_{99}(\text{rel-res}) = 8.194 \times 10^{-1}
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
\langle |\nabla \times B|_r \rangle = 1.919 \times 10^{-19}, \quad\\
\langle |\nabla \times B|_\theta \rangle = 2.621 \times 10^{-19}, \quad\\
\langle |\nabla \times B|_\phi \rangle = 3.975 \times 10^{-19}
$$

$$
\langle |\mu_0 J|_r \rangle = 1.984 \times 10^{-19}, \quad\\
\langle |\mu_0 J|_\theta \rangle = 2.891 \times 10^{-19}, \quad\\
\langle |\mu_0 J|_\phi \rangle = 4.111 \times 10^{-19}
$$

$$
\langle |R|_r \rangle = 1.707 \times 10^{-20}, \quad\\
\langle |R|_\theta \rangle = 8.912 \times 10^{-20}, \quad\\
\langle |R|_\phi \rangle = 4.923 \times 10^{-20}, \quad\\
\langle |R| \rangle = 1.240 \times 10^{-19}
$$

$$
\frac{\langle |\nabla \times B|_r \rangle}{\langle |\mu_0 J|_r \rangle} = 0.9675, \quad
\frac{\langle |\nabla \times B|_\theta \rangle}{\langle |\mu_0 J|_\theta \rangle} = 0.9063, \quad
\frac{\langle |\nabla \times B|_\phi \rangle}{\langle |\mu_0 J|_\phi \rangle} = 0.9670
$$

$$
\text{Residual ratio: } \frac{\| \nabla \times B - \mu_0 J \|}{\| \nabla \times B \| + \| \mu_0 J \|} = 7.087 \times 10^{-2}
$$

$$
\|\nabla \times B\|_\text{total} = 8.515 \times 10^{-19}, \quad\\
\|\mu_0 J\|_\text{total} = 8.986 \times 10^{-19}, \quad\\
\|R\|_\text{total} = 1.240 \times 10^{-19}
$$

$$
\text{Relative residual distribution: } \\
\text{mean(rel-res)} = 1.416 \times 10^{-1}, \quad\\
\text{median(rel-res)} = 1.052 \times 10^{-1}, \quad\\
p_{90}(\text{rel-res}) = 2.670 \times 10^{-1}, \quad\\
p_{99}(\text{rel-res}) = 8.194 \times 10^{-1}
$$


## Equation 6 in static state:

$$
\rho\left(-\Omega_{\text{rot}}\frac{\partial{v_r}}{\partial \phi}+{v} \cdot \nabla {v}\right)=\frac{1}{c} {J} \times {B}-\nabla P+\rho {g}+\nabla \cdot(\nu \rho \nabla {v})
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
&\rho(-\Omega \frac{\partial v_r}{\partial \phi}) &: 3.398 \times 10^{-21} \\
&\rho v_r \frac{\partial v_r}{\partial r} &: 7.401 \times 10^{-21} \\
&\frac{J_r B_r}{c} &: 5.815 \times 10^{-22} \\
&-\frac{\partial p}{\partial r} &: 5.957 \times 10^{-21} \\
&\rho g &: 9.010 \times 10^{-22} \\
&\text{Viscous} &: 3.304 \times 10^{-24} \\
&\text{Residual mean} &: 1.998 \times 10^{-21} \\
&\text{Residual ratio} &: 1.096 \times 10^{-1} \quad 
\end{aligned}
$$

$$
\begin{aligned}
&\text{Relative residual distribution:} \\
&\text{mean(rel-res)} &: 2.035 \times 10^{-1} \\
&\text{median(rel-res)} &: 1.305 \times 10^{-1} \\
&\text{p90(rel-res)} &: 4.566 \times 10^{-1} \\
&\text{p99(rel-res)} &: 1.132 \times 10^{0} \\
&\text{→ Expect: mean(rel-res) ≲ 1e-2 for good physical match.}
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
&\rho(-\omega \frac{\partial v_r}{\partial \phi}) &: 3.398 \times 10^{-20} \\
&\rho v_r \frac{\partial v_r}{\partial r} &: 7.401 \times 10^{-20} \\
&\frac{J_r B_r}{c} &: 1.939 \times 10^{-29} \\
&-\frac{\partial p}{\partial r} &: 5.957 \times 10^{-20} \\
&\rho g &: 9.010 \times 10^{-21} \\
&\text{Viscous} &: 3.304 \times 10^{-23} \\
&\text{Residual mean} &: 1.943 \times 10^{-20} \\
&\text{Residual ratio} &: 1.100 \times 10^{-1} \quad 
\end{aligned}
$$

$$
\begin{aligned}
&\text{Relative residual distribution:} \\
&\text{mean(rel-res)} &: 2.032 \times 10^{-1} \\
&\text{median(rel-res)} &: 1.277 \times 10^{-1} \\
&\text{p90(rel-res)} &: 4.482 \times 10^{-1} \\
&\text{p99(rel-res)} &: 1.100 \times 10^{0} \\
\end{aligned}
$$
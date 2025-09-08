---
layout: default
title: positional embedding, new loss functions pt.2
---

Exp 48 vs. Exp 31

difference is 48 is H1 loss with no powers of 2 (H1-MAE) and with correct spherical terms

#### CR 2271

<img src="resources/week_34/exp_48/h1mae-cr2271.gif">

At latitude = 0 degrees, middle.

<img src="resources/week_34/exp_48/h1mae-cr2271-55.png">


At latitude = 140 degrees. near north pole.

<img src="resources/week_34/exp_48/h1mae-cr2271-10.png">


At latitude = -140 degrees. near south pole.

<img src="resources/week_34/exp_48/h1mae-cr2271-100.png">

#### CR 2289

<img src="resources/week_34/exp_48/h1mae-cr2289.gif">

At latitude = 0 degrees, middle.

<img src="resources/week_34/exp_48/h1mae-cr2289-55.png">


At latitude = 140 degrees. near north pole.

<img src="resources/week_34/exp_48/h1mae-cr2289-10.png">


At latitude = -140 degrees. near south pole.

<img src="resources/week_34/exp_48/h1mae-cr2289-100.png">

#### CR 2291

<img src="resources/week_34/exp_48/h1mae-cr2291.gif">

At latitude = 0 degrees, middle.

<img src="resources/week_34/exp_48/h1mae-cr2291-55.png">


At latitude = 140 degrees. near north pole.

<img src="resources/week_34/exp_48/h1mae-cr2291-10.png">


At latitude = -140 degrees. near south pole.

<img src="resources/week_34/cr2291-100.png">

## Positional Embedding

We were using equiangular grids for the previous papers.

Exp 47 vs. Exp 31

difference is 47 doesnt have the incorrect embedding in 31 (grid_embedding=None)

<img src="resources/week_34/exp_47//cr2289.gif">

Exp 49 vs. Exp 31

difference is 49 has the spherical embedding whereas it is equiangular in 31

TODO

Exp 50 vs. Exp 44

difference is 50 has the spherical embedding + radii whereas its equiangular in 44 with no radii
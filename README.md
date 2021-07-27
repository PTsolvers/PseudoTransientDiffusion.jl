# PseudoTransientDiffusion.jl

This repository contains various diffusion solvers to examplify, test and asses the performance of the pseudo-transient method, including the convergence acceleration via damping (second order Richardson method \[[Frankel, 1950](https://doi.org/10.2307/2002770)\]).

🚧 work in progress - more to come soon.

💡 Link to the [Overleaf draft](https://www.overleaf.com/project/5ff83a57858b372f63143b8e)

## Content
* [The diffusion equation](#the-diffusion-equation)
* [Scripts](#scripts)
* [Results](#results)
* [References](#references)

## The diffusion equation
In this study we will use the (non-linear) diffusion (reaction) equation in 1D, 2D and 3D.

### The 1D diffusion
The linear equation has `D=1` while the nonlinear version considers the case `D=H^3`:
```julia
qHx   = -D*dH/dx
dH/dt = -dqHx/dx
```
### The 2D diffusion
The linear equation has `D=1` while the nonlinear version considers the case `D=H^3`:
```julia
qHx   = -D*dH/dx
qHy   = -D*dH/dy
dH/dt = -(dqHx/dx + dqHy/dy)
```
### The 3D diffusion
The linear equation has `D=1` while the nonlinear version considers the case `D=H^3`:
```julia
qHx   = -D*dH/dx
qHy   = -D*dH/dy
qHz   = -D*dH/dz
dH/dt = -(dqHx/dx + dqHy/dy + dqHz/dz)
```

## Scripts


## Results

## References
[Frankel, S. P. (1950). Convergence rates of iterative treatments of partial differential equations, Mathe. Tables Other Aids Comput., 4, 65–75.](https://doi.org/10.2307/2002770)


[ParallelStencil.jl]: https://github.com/omlins/ParallelStencil.jl

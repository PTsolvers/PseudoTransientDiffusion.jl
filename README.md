# PseudoTransientDiffusion.jl

[![Build Status](https://github.com/PTsolvers/PseudoTransientDiffusion.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PTsolvers/PseudoTransientDiffusion.jl/actions/workflows/CI.yml?query=branch%3Amain)

Parallel (multi-) XPU iterative 1D, 2D and 3D diffusion solvers (linear, step-function and nonlinear diffusion coefficient). This software is part of the [the PTsolvers project](https://ptsolvers.github.io/).

The aim of [the PTsolvers project](https://ptsolvers.github.io/) is to examplify, test and asses the performance of the pseudo-transient method, implementing second-order convergence acceleration building upon the second order Richardson method \[[Frankel, 1950](https://doi.org/10.2307/2002770)\].

> 💡 Link to the [Overleaf draft](https://www.overleaf.com/project/5ff83a57858b372f63143b8e)

## Content
* [Diffusive processes](#diffusive-processes)
* [Scripts](#scripts)
* [Additional infos](#additional-infos)


## Diffusive processes
In this study we will use the (non-linear) diffusion (reaction) equation in 1D, 2D and 3D.

### The diffusion equation
The linear equation has `D=1` while the nonlinear version considers the case `D=H^3`:
```julia
qH    = -D ∇ H
dH/dt = -∇ ⋅ qH
```


## Scripts

### Optimal iteration parameters
The folder [**dispersion_analysis**](/dispersion_analysis) contains the analytical derivations for the values of iteration parameters. We provide these derivations for 1D stationary and transient diffusion problems. Only the case of `D=const` is considered.

The main output of the script is the theoretically predicted value for the non-dimensional parameter `Re`, which is used in the diffusion solvers. The figure showing the dependency of the residual decay rate on `Re` is also displayed:

<img src="dispersion_analysis/fig_dispersion_analysis_transient_diffusion1D.png" alt="Results of the dispersion analysis for the transient diffusion problem" width="500">

For users' convenience, we provide two versions of each script, one version written in Matlab and the other in Python.

To launch the Matlab version, the working installation of Matlab and [Matlab Symbolic Math Toolbox](https://www.mathworks.com/products/symbolic.html) is required.

The second version is implemented using the open-source computer algebra library [SymPy](https://www.sympy.org/) as a [Jupyter](https://jupyter.org/)/[IPython](https://ipython.org/) notebook. The Jupyter notebooks can be viewed directly at GitHub ([example](https://github.com/PTsolvers/PseudoTransientDiffusion.jl/blob/main/dispersion_analysis/dispersion_analysis_transient_diffusion1D.ipynb)). However, in order to view the notebook on a local computer or to make changes to the scripts, the recent Python installation is required. Also, several Python packages need to be installed: SymPy, NumPy, Jupyter, and Matplotlib. The easiest way to install these packages along with their dependencies is to use the [Anaconda](https://www.anaconda.com/products/individual) platform.

After installing `Anaconda`, open the terminal, `cd` into the `dispersion_analysis` folder and create a new `conda` environment with the following command:
```
> conda create -n ptsolvers sympy numpy jupyter matplotlib
```
This command will install the required packages. After the installation completes, activate the environment:
```
> conda activate ptsolvers
```
The final step is to launch the Jupyter server:
```
> jupyter notebook
```
This command starts a server and opens the browser window with file manager.

## Additional infos



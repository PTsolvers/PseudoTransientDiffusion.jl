# PseudoTransientDiffusion.jl

[![Build Status](https://github.com/PTsolvers/PseudoTransientDiffusion.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PTsolvers/PseudoTransientDiffusion.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![DOI](https://zenodo.org/badge/364610588.svg)](https://zenodo.org/badge/latestdoi/364610588)

Parallel (multi-) XPU iterative 1D, 2D and 3D diffusion solvers (resolving linear, step-function and nonlinear diffusion coefficient). This software is part of the [**PTsolvers project**](https://ptsolvers.github.io/).

The aim of this project is to provide iterative solvers **assessing the scalability, performance, and robustness of the accelerated pseudo-transient method** with application to diffusion processes. The solution strategy characterises as semi-iterative, implementing the second-order convergence acceleration as introduced by, e.g., \[[Frankel, 1950](https://doi.org/10.2307/2002770)\].

This repository, together with [**PseudoTransientStokes.jl**](https://github.com/PTsolvers/PseudoTransientStokes.jl/), relates to the original research article draft submitted to the _**Journal XXX**_:
```tex
@article{raess2022,
    title = {{ }},
    journal = {Journal XXX}
}
```

## Content
* [The diffusion equation](#the-diffusion-equation)
* [Scripts](#scripts)
* [Optimal iteration parameters](#optimal-iteration-parameters)
* [Additional infos](#additional-infos)
* [Questions, comments and discussions](#questions-comments-and-discussions)

## The diffusion equation
In this study we will use the (non-linear) diffusion (reaction) equation in 1D, 2D and 3D, solving the following equation:
```julia
qH    = -D ∇ H
dH/dt = -∇ ⋅ qH
```
where `D` stands for the diffusion coefficient, defined as
- `D=1` in the linear case;
- `D=1` or `D=1e-4` in the linear step function case;
- `D=H^3` in the nonlinear (power-law) case.

We use the following initial condition in 1D, 2D, 3D, respectively:
 
<img src="visu/fig_ini.png" alt="Initial conditions for the transient diffusion problem" width="800">

## Scripts
The [**scripts**](/scripts) folder contains the various Julia routines to solve the diffusion equation in 1D (`diff_1D_*.jl`), 2D (`diff_2D_*.jl`) and 3D (`diff_3D_*.jl`). The 3D scripts are grouped in a separate [diff_3D](/scripts/diff_3D) including shell scripts to automatise multi-XPU execution. All Julia routines depend on:
- [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) to allow for backend-agnostic parallelisation on multi-threaded CPUs and Nvidia GPUs (AMD support being _wip_)
- [Plots.jl](https://github.com/JuliaPlots) for basic visualisation
- [MAT.jl](https://github.com/JuliaIO/MAT.jl) for file I/O (disk save for publication-ready figures post-processing using Matlab [visualisation scripts](/visu))

All 3D routines, with exception of one, rely additionally on:
- [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) to implement global grid domain decomposition, relying on 
- [MPI.jl](https://github.com/JuliaParallel/MPI.jl) as communication library for distributed memory, point-to-point communication.

### Running the scripts
To get started, clone or download this repository, launch Julia in project mode `julia --project` and `instantiate` or `resolve` the dependencies from within the REPL in package mode `julia> ]`.

The 1D and 2D scripts can be launched either from with the REPL:
```julia-repl
julia> include("diff_<script_name>.jl")
```
or executed from the shell as:
```shell
julia --project --check-bounds=no -O3 diff_<script_name>.jl
```
Note that for optimal performance, scripts should be launched from the shell making sure bound-checking to be deactivated.

The 3D scripts can be launched in distributed configuration using, e.g., MPI. This requires either to use the Julia MPI launcher `mpiexecjl` or to rely on system provided MPI ([more here](https://juliaparallel.github.io/MPI.jl/latest/configuration/)).

_:bulb: **Note:** refer to the documentation of your supercomputing centre for instructions to run Julia at scale. Instructions for running on the Piz Daint GPU supercomputer at the [Swiss National Supercomputing Centre](https://www.cscs.ch/computers/piz-daint/) can be found [here](https://user.cscs.ch/tools/interactive/julia/)._

All scripts parse environment variables defining important simulation parameters, defaulting to heuristics, namely
```julia
USE_RETURN=false, USE_GPU=false, DO_VIZ=true, DO_SAVE=false, DO_SAVE_VIZ=false
```
and
```julia
NX, NY, NZ
```
defaulting to `512` in 1D and 2D and `64` in 3D. Running, e.g., a script from the shell using the GPU backend can be achieved as following:
```shell
USE_GPU=true julia --project --check-bounds=no -O3 diff_<script_name>.jl
```

## Optimal iteration parameters
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
The repository implements a reference tests suite, using [ReferenceTests.jl](https://github.com/JuliaTesting/ReferenceTests.jl), to verify the correctness of the outputed results with resepct to a reference solution.

## Questions, comments and discussions
To discuss technical issues, please post on Julia Discourse in the [Julia at Scale topic](https://discourse.julialang.org/c/domain/parallel/) or in the `#gpu` or `#distributed` channels on the [Julia Slack](https://julialang.slack.com/) (to join, visit https://julialang.org/slack/).
To discuss numerical/domain-science issues, please post on Julia Discourse in the [Numerics topic](https://discourse.julialang.org/c/domain/numerics/) or the [Modelling & Simulations topic](https://discourse.julialang.org/c/domain/models) or whichever other topic fits best your issue.

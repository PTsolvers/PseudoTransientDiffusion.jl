const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const do_viz  = haskey(ENV, "DO_VIZ")  ? parse(Bool, ENV["DO_VIZ"])  : true
const nx = haskey(ENV, "NX") ? parse(Int, ENV["NX"]) : 512
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_flux!(qHx, qHx2, H, D, θr_θkin, dx)
    @all(qHx)  = (@all(qHx) * θr_θkin - D * @d(H) / dx) / (1.0 + θr_θkin)
    @all(qHx2) = -D * @d(H) / dx
    return
end

@parallel function compute_update!(H, Heq, qHx, θkin_ρ, θkin, dx)
    @inn(H) = (@inn(H) +  θkin_ρ * (@inn(Heq) / θkin - @d(qHx) / dx)) / (1.0 + θkin_ρ / θkin)
    return
end

@parallel function check_res!(ResH, H, Heq, qHx2, θkin, dx)
    @all(ResH) = -(@inn(H) - @inn(Heq)) / θkin - @d(qHx2) / dx
    return
end

@views function diffusion_react_1D()
    # Physics
    lx      = 20.0       # domain size
    D       = 1          # diffusion coefficient
    θkin    = 0.1        # characteristic time of reaction
    # Numerics
    # nx     = 2*256      # numerical grid resolution
    tol     = 1e-8       # tolerance
    itMax   = 1e5        # max number of iterations
    nout    = 10         # tol check
    CFL     = 0.99       # CFL number
    Da      = π + sqrt(π^2 + (lx^2 / D / θkin)) # Numerical Reynolds number
    # Derived numerics
    dx      = lx / nx      # grid size
    Vpdτ    = CFL * dx
    θr_θkin = lx / Vpdτ / Da
    θkin_ρ  = Vpdτ * lx / D / Da
    xc      = LinRange(-lx / 2, lx / 2, nx)
    # Array allocation
    qHx     = @zeros(nx - 1)
    qHx2    = @zeros(nx - 1)
    ResH    = @zeros(nx - 2)
    # Initial condition
    H0     = Data.Array(exp.(-xc.^2))
    Heq    = @ones(nx) .* H0
    H      = @zeros(nx)
    # Time loop
    iter = 0; err = 2 * tol
    # Pseudo-transient iteration
    while err > tol && iter < itMax
        @parallel compute_flux!(qHx, qHx2, H, D, θr_θkin, dx)
        @parallel compute_update!(H, Heq, qHx, θkin_ρ, θkin, dx)
        iter += 1
        if iter % nout == 0
            @parallel check_res!(ResH, H, Heq, qHx2, θkin, dx)
            err = norm(ResH) / sqrt(length(ResH))
        end
    end
    if isnan(err) error("NaN") end
    @printf("nx = %d, iterations tot = %d \n", nx, iter)
    # Visualise
    if do_viz plot(xc, Array(H0), linewidth=3); display(plot!(xc, Array(H), legend=false, framestyle=:box, linewidth=3, xlabel="lx", ylabel="H", title="linear diffusion-reaction (iters=$iter)")) end
    return
end

diffusion_react_1D()

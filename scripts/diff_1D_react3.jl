const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_flux!(qHx, qHx2, H, D, τr_τkin, dx)
    @all(qHx)  = (@all(qHx) * τr_τkin - D * @d(H) / dx) / (1.0 + τr_τkin)
    @all(qHx2) = -D * @d(H) / dx
    return
end

@parallel function compute_update!(H, Heq, qHx, τkin_ρ, τkin, dx)
    @inn(H) = (@inn(H) +  τkin_ρ * (@inn(Heq) / τkin - @d(qHx) / dx)) / (1.0 + τkin_ρ / τkin)
    return
end

@parallel function check_res!(ResH, H, Heq, qHx2, τkin, dx)
    @inn(ResH) = -(@inn(H) - @inn(Heq)) / τkin - @d(qHx2) / dx
    return
end

@views function diffusion_1D(; nx=512, do_viz=false)
    # Physics
    lx      = 20.0       # domain size
    D       = 1          # diffusion coefficient
    τkin    = 0.1        # characteristic time of reaction
    # Numerics
    # nx     = 2*256      # numerical grid resolution
    tol     = 1e-8       # tolerance
    itMax   = 1e5        # max number of iterations
    nout    = 10         # tol check
    CFL     = 1.0        # CFL number
    Da      = π + sqrt(π^2 + (lx^2 / D / τkin)) # Numerical Reynolds number
    # Derived numerics
    dx      = lx / nx      # grid size
    Vpdt    = CFL * dx
    τr_τkin = lx / Vpdt / Da
    τkin_ρ  = Vpdt * lx / D / Da
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
        @parallel compute_flux!(qHx, qHx2, H, D, τr_τkin, dx)
        @parallel compute_update!(H, Heq, qHx, τkin_ρ, τkin, dx)
        iter += 1
        if iter % nout == 0
            @parallel check_res!(ResH, H, Heq, qHx2, τkin, dx)
            err = norm(ResH) / length(ResH)
        end
    end
    if isnan(err) error("NaN") end
    @printf("nx = %d, iterations tot = %d \n", nx, iter)
    # Visualise
    if do_viz plot(xc, Array(H0), linewidth=3); display(plot!(xc, Array(H), legend=false, framestyle=:box, linewidth=3, xlabel="lx", ylabel="H", title="linear diffusion-reaction (iters=$iter)")) end
    return nx, iter
end

diffusion_1D(; do_viz=true)

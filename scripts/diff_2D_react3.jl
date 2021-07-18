const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_flux!(qHx, qHy, qHx2, qHy2, H, D, τr_τkin, dx, dy)
    @all(qHx)  = (@all(qHx) * τr_τkin - D * @d_xi(H) / dx) / (1.0 + τr_τkin)
    @all(qHy)  = (@all(qHy) * τr_τkin - D * @d_yi(H) / dy) / (1.0 + τr_τkin)
    @all(qHx2) = -D * @d_xi(H) / dx
    @all(qHy2) = -D * @d_yi(H) / dy
    return
end

@parallel function compute_update!(H, Heq, qHx, qHy, τkin_ρ, τkin, dx, dy)
    @inn(H) = (@inn(H) +  τkin_ρ * (@inn(Heq) / τkin - (@d_xa(qHx) / dx + @d_ya(qHy) / dy))) / (1.0 + τkin_ρ / τkin)
    return
end

@parallel function check_res!(ResH, H, Heq, qHx2, qHy2, τkin, dx, dy)
    @all(ResH) = -(@inn(H) - @inn(Heq)) / τkin - (@d_xa(qHx2) / dx + @d_ya(qHy2) / dy)
    return
end

@views function diffusion_2D(; nx=512, ny=512, do_viz=false)
    # Physics
    lx, ly  = 10.0, 10.0    # domain size
    D       = 1.0           # diffusion coefficient
    τkin    = 0.1           # characteristic time of reaction
    # Numerics
    # nx     = 2*256        # numerical grid resolution
    tol     = 1e-8          # tolerance
    itMax   = 1e5           # max number of iterations
    nout    = 10            # tol check
    CFL     = 1/sqrt(2)     # CFL number
    Da      = π + sqrt(π^2 + (lx^2 / D / τkin)) # Numerical Reynolds number
    # Derived numerics
    dx, dy  = lx / nx, ly / ny  # grid size    
    Vpdt    = CFL * min(dx, dy)
    τr_τkin = max(lx, ly) / Vpdt / Da
    τkin_ρ  = Vpdt * max(lx, ly) / D / Da
    xc, yc  = LinRange(-lx / 2, lx / 2, nx), LinRange(-ly / 2, ly / 2, ny)
    # Array allocation
    qHx     = @zeros(nx - 1,ny - 2)
    qHy     = @zeros(nx - 2,ny - 1)
    qHx2    = @zeros(nx - 1,ny - 2)
    qHy2    = @zeros(nx - 2,ny - 1)
    ResH    = @zeros(nx - 2,ny - 2)
    # Initial condition
    H0      = Data.Array(exp.(-xc.^2 .- (yc').^2))
    Heq     = @ones(nx,ny) .* H0
    H       = @zeros(nx,ny)
    # Time loop
    iter = 0; err = 2 * tol
    # Pseudo-transient iteration
    while err > tol && iter < itMax
        @parallel compute_flux!(qHx, qHy, qHx2, qHy2, H, D, τr_τkin, dx, dy)
        @parallel compute_update!(H, Heq, qHx, qHy, τkin_ρ, τkin, dx, dy)
        iter += 1
        if iter % nout == 0
            @parallel check_res!(ResH, H, Heq, qHx2, qHy2, τkin, dx, dy)
            err = norm(ResH) / length(ResH)
        end
    end
    if isnan(err) error("NaN") end
    @printf("nx = %d, iterations tot = %d \n", nx, iter)
    # Visualise
    if do_viz display(heatmap(xc, yc, Array(H'), aspect_ratio=1, framestyle=:box, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), xlabel="lx", ylabel="ly", c=:hot, clims=(0, 1), title="linear diffusion-reaction (iters=$iter)")) end
    return nx, ny, iter
end

diffusion_2D(; do_viz=true)

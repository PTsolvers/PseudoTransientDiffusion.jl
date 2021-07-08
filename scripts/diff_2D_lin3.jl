const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_flux!(qHx, qHy, qHx2, qHy2, H, D, τr_dt, dx, dy)
    @all(qHx)  = (@all(qHx) * τr_dt - D * @d_xi(H) / dx) / (1.0 + τr_dt)
    @all(qHy)  = (@all(qHy) * τr_dt - D * @d_yi(H) / dy) / (1.0 + τr_dt)
    @all(qHx2) = -D * @d_xi(H) / dx
    @all(qHy2) = -D * @d_yi(H) / dy
    return
end

@parallel function compute_update!(H, Hold, qHx, qHy, dt_ρ, dt, dx, dy)
    @inn(H) = (@inn(H) +  dt_ρ * (@inn(Hold) / dt - (@d_xa(qHx) / dx + @d_ya(qHy) / dy))) / (1.0 + dt_ρ / dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx2, qHy2, dt, dx, dy)
    @all(ResH) = -(@inn(H) - @inn(Hold)) / dt - (@d_xa(qHx2) / dx + @d_ya(qHy2) / dy)
    return
end

@views function diffusion_2D(; nx=512, ny=512, do_viz=false)
    # Physics
    lx, ly  = 10.0, 10.0    # domain size
    D       = 1.0           # diffusion coefficient
    ttot    = 1.0           # total simulation time
    dt      = 0.2           # physical time step
    # Numerics
    # nx     = 2*256        # numerical grid resolution
    tol     = 1e-8          # tolerance
    itMax   = 1e5           # max number of iterations
    nout    = 10            # tol check
    CFL     = 1/sqrt(2)     # CFL number
    Re      = π + sqrt(π^2 + (lx^2 / D / dt)) # Numerical Reynolds number
    # Derived numerics
    dx, dy  = lx / nx, ly / ny  # grid size    
    Vpdt   = CFL * min(dx, dy)
    τr_dt  = max(lx, ly) / Vpdt / Re
    dt_ρ   = Vpdt * max(lx, ly) / D / Re
    xc, yc  = LinRange(-lx / 2, lx / 2, nx), LinRange(-ly / 2, ly / 2, ny)
    # Array allocation
    qHx     = @zeros(nx - 1,ny - 2)
    qHy     = @zeros(nx - 2,ny - 1)
    qHx2    = @zeros(nx - 1,ny - 2)
    qHy2    = @zeros(nx - 2,ny - 1)
    ResH    = @zeros(nx - 2,ny - 2)
    # Initial condition
    H0      = Data.Array(exp.(-xc.^2 .- (yc').^2))
    Hold    = @ones(nx,ny) .* H0
    H       = @ones(nx,ny) .* H0
    t = 0.0; it = 0; ittot = 0
    # Physical time loop
    while t < ttot
        iter = 0; err = 2 * tol
        # Pseudo-transient iteration
        while err > tol && iter < itMax
            @parallel compute_flux!(qHx, qHy, qHx2, qHy2, H, D, τr_dt, dx, dy)
            @parallel compute_update!(H, Hold, qHx, qHy, dt_ρ, dt, dx, dy)
            iter += 1
            if iter % nout == 0
                @parallel check_res!(ResH, H, Hold, qHx2, qHy2, dt, dx, dy)
                err = norm(ResH) / length(ResH)
            end
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz display(heatmap(xc, yc, Array(H'), aspect_ratio=1, framestyle=:box, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), xlabel="lx", ylabel="ly", c=:hot, clims=(0, 1), title="linear diffusion (nt=$it, iters=$ittot)")) end
    return nx, ny, ittot
end

diffusion_2D(; do_viz=true)

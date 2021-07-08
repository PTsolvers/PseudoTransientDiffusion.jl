const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_Re!(Re, D, lx, dt)
    @inn(Re) = π + sqrt(π^2 + (lx^2 / @maxloc(D) / dt))
    return
end

@parallel function compute_iter_params!(τr_dt, dt_ρ, Re, D, Vpdt, lx)
    @all(τr_dt) = lx / Vpdt / @av(Re)
    @all(dt_ρ)  = Vpdt * lx / @maxloc(D) / @inn(Re)
    return
end

@parallel function compute_flux!(qHx, qHx2, H, D, τr_dt, dx)
    @all(qHx)  = (@all(qHx) * @all(τr_dt) - @av(D) * @d(H) / dx) / (1.0 + @all(τr_dt))
    @all(qHx2) = -@av(D) * @d(H) / dx
    return
end

@parallel function compute_update!(H, Hold, qHx, dt_ρ, dt, dx)
    @inn(H) = (@inn(H) +  @all(dt_ρ) * (@inn(Hold) / dt - @d(qHx) / dx)) / (1.0 + @all(dt_ρ) / dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx2, dt, dx)
    @inn(ResH) = -(@inn(H) - @inn(Hold)) / dt - @d(qHx2) / dx
    return
end

@views function diffusion_1D(; nx=512, do_viz=false)
    # Physics
    lx     = 20.0       # domain size
    D1     = 1.0        # diffusion coefficient
    D2     = 1e-4       # diffusion coefficient
    ttot   = 1.0        # total simulation time
    dt     = 0.1        # physical time step
    # Numerics
    # nx     = 2*256      # numerical grid resolution
    tol    = 1e-8       # tolerance
    itMax  = 1e5        # max number of iterations
    nout   = 10         # tol check
    CFL    = 1.0        # CFL number
    # Derived numerics
    dx     = lx / nx      # grid size
    Vpdt   = CFL * dx
    xc     = LinRange(-lx / 2, lx / 2, nx)
    # Array allocation
    qHx    = @zeros(nx - 1)
    qHx2   = @zeros(nx - 1)
    ResH   = @zeros(nx - 2)
    Re     = @zeros(nx  )
    τr_dt  = @zeros(nx - 1)
    dt_ρ   = @zeros(nx - 2)
    # Initial condition
    D      = D1 * @ones(nx)
    D[1:Int(ceil(nx / 2.2))] .= D2
    H0     = Data.Array(exp.(-xc.^2))
    Hold   = @ones(nx) .* H0
    H      = @ones(nx) .* H0
    @parallel compute_Re!(Re, D, lx, dt)
    Re[1] = Re[2]; Re[end] = Re[end-1]
    @parallel compute_iter_params!(τr_dt, dt_ρ, Re, D, Vpdt, lx)
    t = 0.0; it = 0; ittot = 0; nt = ceil(ttot / dt)
    # Physical time loop
    while it < nt
        iter = 0; err = 2 * tol
        # Pseudo-transient iteration
        while err > tol && iter < itMax
            @parallel compute_flux!(qHx, qHx2, H, D, τr_dt, dx)
            @parallel compute_update!(H, Hold, qHx, dt_ρ, dt, dx)
            iter += 1
            if iter % nout == 0
                @parallel check_res!(ResH, H, Hold, qHx2, dt, dx)
                err = norm(ResH) / length(ResH)
            end
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz plot(xc, Array(H0), linewidth=3); display(plot!(xc, Array(H), legend=false, framestyle=:box, linewidth=3, xlabel="lx", ylabel="H", title="linear step diffusion (nt=$it, iters=$ittot)")) end
    return nx, ittot
end

diffusion_1D(; do_viz=true)

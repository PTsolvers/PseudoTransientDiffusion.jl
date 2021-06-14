const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_flux1!(qHx, qHx2, H, D, dx)
    @all(qHx)  = -D * @d(H) / dx
    @all(qHx2) = @all(qHx)
    return
end

@parallel function compute_update1!(H, Hold, qHx, dH, ρ, τr, dt, dx)
    @all(dH) = (@all(dH) * ρ * τr - @d(qHx) / dx - (@inn(H) - @inn(Hold)) / dt) / (ρ * τr + τr / dt + ρ)
    @inn(H)  = @inn(H) + @all(dH)
    return
end

@parallel function compute_flux2!(qHx, qHx2, H, D, τr, dx)
    @all(qHx)  = (@all(qHx) * τr - D * @d(H) / dx) / (1.0 + τr)
    @all(qHx2) = -D * @d(H) / dx
    return
end

@parallel function compute_update2!(H, Hold, qHx, ρ, dt, dx)
    @inn(H) = (@inn(H) * ρ +  @inn(Hold) / dt - @d(qHx) / dx) / (ρ + 1 / dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx2, dt, dx)
    @inn(ResH) = -(@inn(H) - @inn(Hold)) / dt - @d(qHx2) / dx
    return
end

@views function diffusion_1D(; nx=512, do_viz=false)
    # Physics
    lx     = 20.0       # domain size
    D      = 1          # diffusion coefficient
    ttot   = 1          # total simulation time
    dt     = 0.1        # physical time step
    # Numerics
    # nx     = 2*256      # numerical grid resolution
    tol    = 1e-8       # tolerance
    itMax  = 1e5        # max number of iterations
    nout   = 10         # tol check
    accel  = 2
    # Derived numerics
    dx     = lx / nx      # grid size
    CFLdx  = 1.0 * dx
    dmp    = π + sqrt(π^2 + (lx^2 / D / dt))
    τr     = lx / CFLdx / dmp;
    ρ      = D * dmp / lx / CFLdx;
    xc     = LinRange(-lx / 2, lx / 2, nx)
    # Array allocation
    qHx    = @zeros(nx - 1)
    qHx2   = @zeros(nx - 1)
    ResH   = @zeros(nx - 2)
    dH     = @zeros(nx - 2);
    # Initial condition
    H0     = Data.Array(exp.(-xc.^2 / D))
    Hold   = @ones(nx) .* H0
    H      = @ones(nx) .* H0
    t = 0.0; it = 0; ittot = 0
    # Physical time loop
    while t < ttot
        iter = 0; err = 2 * tol
        # Pseudo-transient iteration
        while err > tol && iter < itMax
            if accel == 1
                @parallel compute_flux1!(qHx, qHx2, H, D, dx)
                @parallel compute_update1!(H, Hold, qHx, dH, ρ, τr, dt, dx)
            else
                @parallel compute_flux2!(qHx, qHx2, H, D, τr, dx)
                @parallel compute_update2!(H, Hold, qHx, ρ, dt, dx)
            end
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
    # Analytic solution
    Hana = 1 / sqrt(4 * (ttot + 1 / 4)) * exp.(-xc.^2 / (4 * D * (ttot + 1 / 4)))
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d, error vs analytic = %1.2e \n", round(ttot, sigdigits=2), it, nx, ittot, norm(Array(H) - Hana) / sqrt(nx))
    # Visualise
    if do_viz plot(xc, Array(H0), linewidth=3); display(plot!(xc, [Array(H) Array(Hana)], legend=false, framestyle=:box, linewidth=3, xlabel="lx", ylabel="H", title="linear diffusion (nt=$it, iters=$ittot)")) end
    return nx, ittot
end

# diffusion_1D(; do_viz=true)

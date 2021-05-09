const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end
using Plots, Printf, LinearAlgebra

macro H3() esc(:( @av(H)*@av(H)*@av(H) )) end
@parallel function compute_flux!(qHx, H, dx)
    @all(qHx) = -@H3()*@d(H)/dx
    return
end

@parallel function compute_rate!(ResH, dHdt, H, Hold, qHx, dt, damp, dx)
    @all(ResH) = -(@inn(H) - @inn(Hold))/dt -@d(qHx)/dx
    @all(dHdt) = @all(ResH) + damp*@all(dHdt)
    return
end

macro dtau() esc(:( 1.0/(1.0/(dx*dx/@H3()/2.1) + 1.0/dt) )) end
@parallel function compute_update!(H, dHdt, dt, dx)
    @inn(H) = @inn(H) + @dtau()*@all(dHdt)
    return
end

@views function diffusion_1D(; nx=512, do_viz=false)
    # Physics
    lx     = 10.0       # domain size
    ttot   = 1.0        # total simulation time
    dt     = 0.2        # physical time step
    # Numerics
    # nx     = 2*256        # numerical grid resolution
    tol    = 1e-6       # tolerance
    itMax  = 1e5        # max number of iterations
    damp   = 1-min(nx,44)/nx    # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    # Derived numerics
    dx     = lx/nx      # grid size
    xc     = LinRange(dx/2, lx-dx/2, nx)
    # Array allocation
    qHx    = @zeros(nx-1)
    dHdt   = @zeros(nx-2)
    ResH   = @zeros(nx-2)
    # Initial condition
    H0     = Data.Array( exp.(-(xc.-lx/2).^2) )
    Hold   = @ones(nx).*H0
    H      = @ones(nx).*H0
    t = 0.0; it = 0; ittot = 0
    # Physical time loop
    while t<ttot
        iter = 0; err = 2*tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax
            @parallel compute_flux!(qHx, H, dx)
            @parallel compute_rate!(ResH, dHdt, H, Hold, qHx, dt, damp, dx)
            @parallel compute_update!(H, dHdt, dt, dx)
            iter += 1; err = norm(ResH)/length(ResH)
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz plot(xc, Array(H0), linewidth=3); display(plot!(xc, Array(H), legend=false, framestyle=:box, linewidth=3, xlabel="lx", ylabel="H", title="nonlinear diffusion (nt=$it, iters=$ittot)")) end
    return nx, ittot
end

# diffusion_1D(; )

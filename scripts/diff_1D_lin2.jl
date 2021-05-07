const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_flux!(qHx, H, D, dtauq, dx)
    @all(qHx) = (@all(qHx) - dtauq*@d(H)/dx)/(1.0 + dtauq/D)
    return
end

@parallel function compute_update!(H, Hold, qHx, dtauH, dt, dx)
    @inn(H) = (@inn(H) + dtauH*(@inn(Hold)/dt - @d(qHx)/dx))/(1.0 + dtauH/dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx, dt, dx)
    @inn(ResH) = -(@inn(H)-@inn(Hold))/dt - @d(qHx)/dx
    return
end

@views function diffusion_1D(; nx=512, do_viz=false)
    # Physics
    lx     = 10.0       # domain size
    D      = 1.0        # diffusion coefficient
    ttot   = 1.0        # total simulation time
    dt     = 0.2        # physical time step
    # Numerics
    # nx     = 2*256      # numerical grid resolution
    tol    = 1e-6       # tolerance
    itMax  = 1e5        # max number of iterations
    # Derived numerics
    dx     = lx/nx      # grid size
    dmp    = 3.0
    CFLdx  = 0.7*dx
    Re_opt = π + sqrt(π^2 + (lx/D)^2)
    dtauq  = dmp*CFLdx*lx/Re_opt
    dtauH  = CFLdx^2/dtauq # dtauH*dtauq = CFL^2*dx^2 -> dt < CFL*dx/Vsound
    xc     = LinRange(dx/2, lx-dx/2, nx)
    # Array allocation
    qHx    = @zeros(nx-1)
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
            @parallel compute_flux!(qHx, H, D, dtauq, dx)
            @parallel compute_update!(H, Hold, qHx, dtauH, dt, dx)
            @parallel check_res!(ResH, H, Hold, qHx, dt, dx)
            iter += 1; err = norm(ResH)/length(ResH)
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
    end
    # Analytic solution
    Hana = 1/sqrt(4*(ttot+1/4)) * exp.(-(xc.-lx/2).^2 /(4*(ttot+1/4)))
    @printf("Total time = %1.2f, time steps = %d, iterations tot = %d, error vs analytic = %1.2e \n", round(ttot, sigdigits=2), it, ittot, norm(Array(H)-Hana))
    # Visualise
    if do_viz plot(xc, Array(H0), linewidth=3); display(plot!(xc, Array(H), legend=false, framestyle=:box, linewidth=3, xlabel="lx", ylabel="H", title="linear diffusion (nt=$it, iters=$ittot)")) end
    return nx, ittot
end

# diffusion_1D(; )

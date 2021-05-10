const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_Re!(Re_opt, D, lx)
    # @inn(Re_opt) = π + sqrt(π^2 + (lx/@inn(D))^2)
    @inn(Re_opt) = π + sqrt(π^2 + (lx/@maxloc(D))^2)
    return
end

@parallel function compute_dtauq!(dtauq, Re_opt, dmp, CFLdx, lx)
    @all(dtauq)  = dmp*CFLdx*lx/@av(Re_opt)
    return
end

@parallel function compute_dtauH!(dtauH, dtauq, CFLdx)
    @all(dtauH)  = CFLdx^2/@av(dtauq) # dtauH*dtauq = CFL^2*dx^2 -> dt < CFL*dx/Vsound
    return
end

@parallel function compute_flux!(qHx, H, D, dtauq, dx)
    @all(qHx) = (@all(qHx) - @all(dtauq)*@d(H)/dx)/(1.0 + @all(dtauq)/@av(D))
    return
end

@parallel function compute_update!(H, Hold, qHx, dtauH, dt, dx)
    @inn(H) = (@inn(H) + @all(dtauH)*(@inn(Hold)/dt - @d(qHx)/dx))/(1.0 + @all(dtauH)/dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx, dt, dx)
    @inn(ResH) = -(@inn(H)-@inn(Hold))/dt - @d(qHx)/dx
    return
end

@views function diffusion_1D(; nx=512, do_viz=false)
    # Physics
    lx     = 10.0       # domain size
    D1     = 1.0        # diffusion coefficient
    D2     = 1e-4       # diffusion coefficient
    ttot   = 1.0        # total simulation time
    dt     = 0.2        # physical time step
    # Numerics
    # nx     = 2*256      # numerical grid resolution
    tol    = 1e-6       # tolerance
    itMax  = 1e5        # max number of iterations
    # Derived numerics
    dx     = lx/nx      # grid size
    dmp    = 1.9
    CFLdx  = 0.7*dx
    xc     = LinRange(dx/2, lx-dx/2, nx)
    # Array allocation
    qHx    = @zeros(nx-1)
    ResH   = @zeros(nx-2)
    Re_opt = @zeros(nx  )
    dtauq  = @zeros(nx-1)
    dtauH  = @zeros(nx-2)
    # Initial condition
    D      = D2*@ones(nx)
    D[1:Int(ceil(nx/2.5))] .= D1
    H0     = Data.Array( exp.(-(xc.-lx/2).^2) )
    Hold   = @ones(nx).*H0
    H      = @ones(nx).*H0
    @parallel compute_Re!(Re_opt, D, lx)
    Re_opt[1] = Re_opt[2]; Re_opt[end] = Re_opt[end-1]
    @parallel compute_dtauq!(dtauq, Re_opt, dmp, CFLdx, lx)
    @parallel compute_dtauH!(dtauH, dtauq, CFLdx)
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
        if isnan(err) error("NaN") end
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz plot(xc, Array(H0), linewidth=3); display(plot!(xc, Array(H), legend=false, framestyle=:box, linewidth=3, xlabel="lx", ylabel="H", title="linear diffusion (nt=$it, iters=$ittot)")) end
    return nx, ittot
end

# diffusion_1D(; do_viz=true)

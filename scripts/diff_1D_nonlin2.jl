const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end
using Plots, Printf, LinearAlgebra

macro innH3()    esc(:( @inn(H)*@inn(H)*@inn(H)          )) end
macro avH3()     esc(:( @av(H)*@av(H)*@av(H)             )) end
macro avRe_opt() esc(:( π + sqrt(π^2 + (lx/@avH3())^2)   )) end
macro Re_opt()   esc(:( π + sqrt(π^2 + (lx/@innH3())^2)  )) end
macro dtauq()    esc(:( dmp*CFLdx*lx/@avRe_opt()         )) end
macro dtauH()    esc(:( CFLdx^2/(dmp*CFLdx*lx/@Re_opt()) )) end # dtauH*dtauq = CFL^2*dx^2 -> dt < CFL*dx/Vsound

@parallel function compute_flux!(qHx, H, dmp, CFLdx, lx, dx)
    @all(qHx) = (@all(qHx) - @dtauq()*@d(H)/dx)/(1.0 + @dtauq()/@avH3())
    return
end

@parallel function compute_update!(H, Hold, qHx, dt, dmp, CFLdx, lx, dx)
    @inn(H) = (@inn(H) + @dtauH()*(@inn(Hold)/dt - @d(qHx)/dx))/(1.0 + @dtauH()/dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx, dt, dx)
    @inn(ResH) = -(@inn(H)-@inn(Hold))/dt - @d(qHx)/dx
    return
end

@views function diffusion_1D(; nx=512, do_viz=false)
    # Physics
    lx     = 10.0       # domain size
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
            @parallel compute_flux!(qHx, H, dmp, CFLdx, lx, dx)
            @parallel compute_update!(H, Hold, qHx, dt, dmp, CFLdx, lx, dx)
            @parallel check_res!(ResH, H, Hold, qHx, dt, dx)
            iter += 1; err = norm(ResH)/length(ResH)
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz plot(xc, Array(H0), linewidth=3); display(plot!(xc, Array(H), legend=false, framestyle=:box, linewidth=3, xlabel="lx", ylabel="H", title="nonlinear diffusion (nt=$it, iters=$ittot)")) end
    return nx, ittot
end

# diffusion_1D(; do_viz=true)

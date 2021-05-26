const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_flux!(qHx, qHy, H, D, dx, dy)
    @all(qHx) = -D*@d_xi(H)/dx
    @all(qHy) = -D*@d_yi(H)/dy
    return
end

@parallel function compute_rate!(ResH, dHdt, H, Hold, qHx, qHy, dt, damp, dx, dy)
    @all(ResH) = -(@inn(H) - @inn(Hold))/dt - (@d_xa(qHx)/dx + @d_ya(qHy)/dy)
    @all(dHdt) = @all(ResH) + damp*@all(dHdt)
    return
end

@parallel function compute_update!(H, dHdt, dtau)
    @inn(H) = @inn(H) + dtau*@all(dHdt)
    return
end

@views function diffusion_2D(; nx=512, ny=512, do_viz=false)
    # Physics
    lx, ly  = 10.0, 10.0   # domain size
    D       = 1.0          # diffusion coefficient
    ttot    = 1.0          # total simulation time
    dt      = 0.2          # physical time step
    # Numerics
    # nx     = 2*256       # numerical grid resolution
    tol     = 1e-8         # tolerance
    itMax   = 1e5          # max number of iterations
    nout    = 10           # tolerance check
    damp    = 1-22/nx      # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    # Derived numerics
    dx, dy  = lx/nx, ly/ny # grid size
    dtau    = (1.0/(min(dx,dy)^2/D/4.1) + 1.0/dt)^-1 # iterative timestep
    xc, yc  = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
    # Array allocation
    qHx     = @zeros(nx-1,ny-2)
    qHy     = @zeros(nx-2,ny-1)
    dHdt    = @zeros(nx-2,ny-2)
    ResH    = @zeros(nx-2,ny-2)
    # Initial condition
    H0      = Data.Array( exp.(-(xc.-lx/2).^2 .-(yc'.-ly/2).^2) )
    Hold    = @ones(nx,ny).*H0
    H       = @ones(nx,ny).*H0
    t = 0.0; it = 0; ittot = 0
    # Physical time loop
    while t<ttot
        iter = 0; err = 2*tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax
            @parallel compute_flux!(qHx, qHy, H, D, dx, dy)
            @parallel compute_rate!(ResH, dHdt, H, Hold, qHx, qHy, dt, damp, dx, dy)
            @parallel compute_update!(H, dHdt, dtau)
            iter += 1; if (iter % nout == 0)  err = norm(ResH)/length(ResH)  end
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz display(heatmap(xc, yc, Array(H'), aspect_ratio=1, framestyle=:box, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), xlabel="lx", ylabel="ly", c=:hot, clims=(0,1), title="linear diffusion (nt=$it, iters=$ittot)")) end
    return nx, ny, ittot
end

# diffusion_2D(; nx=32, ny=32, do_viz=true)

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_dtau!(dtau, D, dt, dx, dy)
    # @all(dtau) = 1.0./(1.0./(dx^2 ./@inn(D)/4.1) .+ 1.0/dt)
    @all(dtau) = 1.0./(1.0./(min(dx,dy)^2 ./@maxloc(D)/4.1) .+ 1.0/dt)
    return
end

@parallel function compute_flux!(qHx, qHy, H, D, dx, dy)
    @all(qHx) = -@av_xi(D)*@d_xi(H)/dx
    @all(qHy) = -@av_yi(D)*@d_yi(H)/dy
    return
end

@parallel function compute_rate!(ResH, dHdt, H, Hold, qHx, qHy, dt, damp, dx, dy)
    @all(ResH) = -(@inn(H) - @inn(Hold))/dt - (@d_xa(qHx)/dx + @d_ya(qHy)/dy)
    @all(dHdt) = @all(ResH) + damp*@all(dHdt)
    return
end

@parallel function compute_update!(H, dHdt, dtau)
    @inn(H) = @inn(H) + @all(dtau)*@all(dHdt)
    return
end

@views function diffusion_2D(; nx=512, ny=512, do_viz=false)
    # Physics
    lx, ly  = 10.0, 10.0   # domain size
    D1      = 1.0          # diffusion coefficient
    D2      = 1e-4         # diffusion coefficient
    ttot    = 1.0          # total simulation time
    dt      = 0.2          # physical time step
    # Numerics
    # nx     = 2*256       # numerical grid resolution
    tol     = 1e-6         # tolerance
    itMax   = 1e5          # max number of iterations
    damp    = 1-22/nx      # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    # Derived numerics
    dx, dy  = lx/nx, ly/ny # grid size
    xc, yc  = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
    # Array allocation
    qHx     = @zeros(nx-1,ny-2)
    qHy     = @zeros(nx-2,ny-1)
    dHdt    = @zeros(nx-2,ny-2)
    ResH    = @zeros(nx-2,ny-2)
    dtau    = @zeros(nx-2,ny-2)
    # Initial condition
    D       = D2*@ones(nx,ny)
    D[1:Int(ceil(nx/2.5)),:] .= D1
    H0      = Data.Array( exp.(-(xc.-lx/2).^2 .-(yc'.-ly/2).^2) )
    Hold    = @ones(nx,ny).*H0
    H       = @ones(nx,ny).*H0
    @parallel compute_dtau!(dtau, D, dt, dx, dy)
    t = 0.0; it = 0; ittot = 0
    # Physical time loop
    while t<ttot
        iter = 0; err = 2*tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax
            @parallel compute_flux!(qHx, qHy, H, D, dx, dy)
            @parallel compute_rate!(ResH, dHdt, H, Hold, qHx, qHy, dt, damp, dx, dy)
            @parallel compute_update!(H, dHdt, dtau)
            iter += 1; err = norm(ResH)/length(ResH)
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz display(heatmap(xc, yc, H', aspect_ratio=1, framestyle=:box, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), xlabel="lx", ylabel="ly", c=:hot, clims=(0,1), title="linear diffusion (nt=$it, iters=$ittot)")) end
    return nx, ny, ittot
end

# diffusion_2D(; do_viz=true)

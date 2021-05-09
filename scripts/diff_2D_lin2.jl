const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_flux!(qHx, qHy, H, D, dtauq, dx, dy)
    @all(qHx) = (@all(qHx) - dtauq*@d_xi(H)/dx)/(1.0 + dtauq/D)
    @all(qHy) = (@all(qHy) - dtauq*@d_yi(H)/dy)/(1.0 + dtauq/D)
    return
end

@parallel function compute_update!(H, Hold, qHx, qHy, dtauH, dt, dx, dy)
    @inn(H) = (@inn(H) + dtauH*(@inn(Hold)/dt - (@d_xa(qHx)/dx + @d_ya(qHy)/dy)))/(1.0 + dtauH/dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx, qHy, dt, dx, dy)
    @inn(ResH) = -(@inn(H)-@inn(Hold))/dt - (@d_xa(qHx)/dx + @d_ya(qHy)/dy)
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
    tol     = 1e-6          # tolerance
    itMax   = 1e5           # max number of iterations
    # Derived numerics
    dx, dy  = lx/nx, ly/ny  # grid size    
    dmp     = 1.9
    CFLdx   = 0.7*dx
    Re_opt  = π + sqrt(π^2 + (lx/D)^2)
    dtauq   = dmp*CFLdx*lx/Re_opt
    dtauH  = CFLdx^2/dtauq # dtauH*dtauq = CFL^2*dx^2 -> dt < CFL*dx/Vsound
    xc     = LinRange(dx/2, lx-dx/2, nx)
    yc     = LinRange(dy/2, ly-dy/2, ny)
    # Array allocation
    qHx    = @zeros(nx-1,ny-2)
    qHy    = @zeros(nx-2,ny-1)
    dHdt   = @zeros(nx-2,ny-2)
    ResH   = @zeros(nx-2,ny-2)
    # Initial condition
    H0     = Data.Array( exp.(-(xc.-lx/2).^2 .-(yc'.-ly/2).^2) )
    Hold   = @ones(nx,ny).*H0
    H      = @ones(nx,ny).*H0
    t = 0.0; it = 0; ittot = 0
    # Physical time loop
    while t<ttot
        iter = 0; err = 2*tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax
            @parallel compute_flux!(qHx, qHy, H, D, dtauq, dx, dy)
            @parallel compute_update!(H, Hold, qHx, qHy, dtauH, dt, dx, dy)
            @parallel check_res!(ResH, H, Hold, qHx, qHy, dt, dx, dy)
            iter += 1; err = norm(ResH)/length(ResH)
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz display(heatmap(xc, yc, H', aspect_ratio=1, framestyle=:box, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), xlabel="lx", ylabel="ly", c=:hot, clims=(0,1), title="linear diffusion (nt=$it, iters=$ittot)")) end
    return nx, ny, ittot
end

# diffusion_2D(; )

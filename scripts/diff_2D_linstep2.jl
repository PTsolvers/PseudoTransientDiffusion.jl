const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
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

@parallel function compute_flux!(qHx, qHy, H, D, dtauq, dx, dy)
    @all(qHx) = (@all(qHx) - @av_ya(dtauq)*@d_xi(H)/dx)/(1.0 + @av_ya(dtauq)/@av_xi(D))
    @all(qHy) = (@all(qHy) - @av_xa(dtauq)*@d_yi(H)/dy)/(1.0 + @av_xa(dtauq)/@av_yi(D))
    return
end

@parallel function compute_update!(H, Hold, qHx, qHy, dtauH, dt, dx, dy)
    @inn(H) = (@inn(H) + @all(dtauH)*(@inn(Hold)/dt - (@d_xa(qHx)/dx + @d_ya(qHy)/dy)))/(1.0 + @all(dtauH)/dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx, qHy, dt, dx, dy)
    @all(ResH) = -(@inn(H)-@inn(Hold))/dt - (@d_xa(qHx)/dx + @d_ya(qHy)/dy)
    return
end

@views function diffusion_2D(; nx=512, ny=512, do_viz=false)
    # Physics
    lx, ly  = 10.0, 10.0    # domain size
    D1      = 1.0          # diffusion coefficient
    D2      = 1e-4         # diffusion coefficient
    ttot    = 1.0           # total simulation time
    dt      = 0.2           # physical time step
    # Numerics
    # nx     = 2*256        # numerical grid resolution
    tol     = 1e-6          # tolerance
    itMax   = 1e5           # max number of iterations
    # Derived numerics
    dx, dy  = lx/nx, ly/ny  # grid size    
    dmp     = 1.2
    CFLdx   = 0.7*dx
    xc, yc  = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
    # Array allocation
    qHx     = @zeros(nx-1,ny-2)
    qHy     = @zeros(nx-2,ny-1)
    ResH    = @zeros(nx-2,ny-2)
    Re_opt  = @zeros(nx  ,ny  )
    dtauq   = @zeros(nx-1,ny-1)
    dtauH   = @zeros(nx-2,ny-2)
    # Initial condition
    D       = D2*@ones(nx,ny)
    D[1:Int(ceil(nx/2.5)),:] .= D1
    H0      = Data.Array( exp.(-(xc.-lx/2).^2 .-(yc'.-ly/2).^2) )
    Hold    = @ones(nx,ny).*H0
    H       = @ones(nx,ny).*H0
    @parallel compute_Re!(Re_opt, D, lx)
    Re_opt[1,:] = Re_opt[2,:]; Re_opt[end,:] = Re_opt[end-1,:]
    Re_opt[:,1] = Re_opt[:,2]; Re_opt[:,end] = Re_opt[:,end-1]
    @parallel compute_dtauq!(dtauq, Re_opt, dmp, CFLdx, lx)
    @parallel compute_dtauH!(dtauH, dtauq, CFLdx)
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
        if isnan(err) error("NaN") end
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz display(heatmap(xc, yc, H', aspect_ratio=1, framestyle=:box, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), xlabel="lx", ylabel="ly", c=:hot, clims=(0,1), title="linear diffusion (nt=$it, iters=$ittot)")) end
    return nx, ny, ittot
end

# diffusion_2D(; do_viz=true)

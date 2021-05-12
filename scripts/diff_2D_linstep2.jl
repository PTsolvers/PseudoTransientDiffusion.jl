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

@parallel function compute_flux!(qHx, qHy, qHx2, qHy2, H, D, dtauq, dx, dy)
    @all(qHx)  = (@all(qHx) - @av_ya(dtauq)*@d_xi(H)/dx)/(1.0 + @av_ya(dtauq)/@av_xi(D))
    @all(qHy)  = (@all(qHy) - @av_xa(dtauq)*@d_yi(H)/dy)/(1.0 + @av_xa(dtauq)/@av_yi(D))
    @all(qHx2) = -@av_xi(D)*@d_xi(H)/dx
    @all(qHy2) = -@av_yi(D)*@d_yi(H)/dy
    return
end

@parallel function compute_update!(H, Hold, qHx, qHy, dtauH, dt, dx, dy)
    @inn(H) = (@inn(H) + @all(dtauH)*(@inn(Hold)/dt - (@d_xa(qHx)/dx + @d_ya(qHy)/dy)))/(1.0 + @all(dtauH)/dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx2, qHy2, dt, dx, dy)
    @all(ResH) = -(@inn(H)-@inn(Hold))/dt - (@d_xa(qHx2)/dx + @d_ya(qHy2)/dy)
    return
end

@views function diffusion_2D(; nx=512, ny=512, do_viz=false)
    # Physics
    lx, ly  = 10.0, 10.0    # domain size
    D1      = 1.0           # diffusion coefficient
    D2      = 1e-4          # diffusion coefficient
    ttot    = 1.0           # total simulation time
    dt      = 0.2           # physical time step
    # Numerics
    # nx     = 2*256        # numerical grid resolution
    tol     = 1e-8          # tolerance
    itMax   = 1e5           # max number of iterations
    nout    = 10            # tol check
    # Derived numerics
    dx, dy  = lx/nx, ly/ny  # grid size    
    dmp     = 2.0
    CFLdx   = 0.7*dx
    xc, yc  = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
    # Array allocation
    qHx     = @zeros(nx-1,ny-2)
    qHy     = @zeros(nx-2,ny-1)
    qHx2    = @zeros(nx-1,ny-2)
    qHy2    = @zeros(nx-2,ny-1)
    ResH    = @zeros(nx-2,ny-2)
    Re_opt  = @zeros(nx  ,ny  )
    dtauq   = @zeros(nx-1,ny-1)
    dtauH   = @zeros(nx-2,ny-2)
    # Initial condition
    D       = D1*@ones(nx,ny)
    D[1:Int(ceil(nx/2.2)),:] .= D2
    D[:,1:Int(ceil(ny/2.2))] .= D2
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
            @parallel compute_flux!(qHx, qHy, qHx2, qHy2, H, D, dtauq, dx, dy)
            @parallel compute_update!(H, Hold, qHx, qHy, dtauH, dt, dx, dy)
            iter += 1
            if iter % nout == 0
                @parallel check_res!(ResH, H, Hold, qHx2, qHy2, dt, dx, dy)
                err = norm(ResH)/length(ResH)
            end
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz display(heatmap(xc, yc, Array(H'), aspect_ratio=1, framestyle=:box, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), xlabel="lx", ylabel="ly", c=:hot, clims=(0,1), title="linear step diffusion (nt=$it, iters=$ittot)")) end
    return nx, ny, ittot
end

# diffusion_2D(; do_viz=true)

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, LinearAlgebra

macro innH3()   esc(:( @inn(H)*@inn(H)*@inn(H) )) end
macro av_xiH3() esc(:( @av_xi(H)*@av_xi(H)*@av_xi(H) )) end
macro av_yiH3() esc(:( @av_yi(H)*@av_yi(H)*@av_yi(H) )) end
macro av_xaH3() esc(:( @av_xa(H)*@av_xa(H)*@av_xa(H) )) end
macro av_yaH3() esc(:( @av_ya(H)*@av_ya(H)*@av_ya(H) )) end
macro Re_opt()  esc(:( π + sqrt(π^2 + (lx/@innH3())^2) )) end
macro av_ya_Re_opt() esc(:( π + sqrt(π^2 + (lx/@av_yaH3())^2) )) end
macro av_xa_Re_opt() esc(:( π + sqrt(π^2 + (lx/@av_xaH3())^2) )) end
macro av_ya_dtauq()  esc(:( dmp*CFLdx*lx/@av_ya_Re_opt() )) end
macro av_xa_dtauq()  esc(:( dmp*CFLdx*lx/@av_xa_Re_opt() )) end
macro dtauH()   esc(:( CFLdx^2/(dmp*CFLdx*lx/@Re_opt())  )) end # dtauH*dtauq = CFL^2*dx^2 -> dt < CFL*dx/Vsound

@parallel function compute_flux!(qHx, qHy, qHx2, qHy2, H, dmp, CFLdx, lx, dx, dy)
    @all(qHx)  = (@all(qHx) - @av_ya_dtauq()*@d_xi(H)/dx)/(1.0 + @av_ya_dtauq()/@av_xiH3())
    @all(qHy)  = (@all(qHy) - @av_xa_dtauq()*@d_yi(H)/dy)/(1.0 + @av_xa_dtauq()/@av_yiH3())
    @all(qHx2) = -@av_xiH3()*@d_xi(H)/dx
    @all(qHy2) = -@av_yiH3()*@d_yi(H)/dy
    return
end

@parallel function compute_update!(H, Hold, qHx, qHy, dt, dmp, CFLdx, lx, dx, dy)
    @inn(H) = (@inn(H) + @dtauH()*(@inn(Hold)/dt - (@d_xa(qHx)/dx + @d_ya(qHy)/dy)))/(1.0 + @dtauH()/dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx2, qHy2, dt, dx, dy)
    @all(ResH) = -(@inn(H)-@inn(Hold))/dt - (@d_xa(qHx2)/dx + @d_ya(qHy2)/dy)
    return
end

@views function diffusion_2D(; nx=512, ny=512, do_viz=false)
    # Physics
    lx, ly  = 10.0, 10.0    # domain size
    ttot    = 1.0           # total simulation time
    dt      = 0.2           # physical time step
    # Numerics
    # nx     = 2*256        # numerical grid resolution
    tol     = 1e-8          # tolerance
    itMax   = 1e5           # max number of iterations
    nout    = 10            # tol check
    # Derived numerics
    dx, dy  = lx/nx, ly/ny  # grid size    
    dmp     = 7.0
    CFLdx   = 0.7*dx
    xc, yc  = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
    # Array allocation
    qHx     = @zeros(nx-1,ny-2)
    qHy     = @zeros(nx-2,ny-1)
    qHx2    = @zeros(nx-1,ny-2)
    qHy2    = @zeros(nx-2,ny-1)
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
            @parallel compute_flux!(qHx, qHy, qHx2, qHy2, H, dmp, CFLdx, lx, dx, dy)
            @parallel compute_update!(H, Hold, qHx, qHy, dt, dmp, CFLdx, lx, dx, dy)
            @parallel check_res!(ResH, H, Hold, qHx2, qHy2, dt, dx, dy)
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

# diffusion_2D(; do_viz=true)

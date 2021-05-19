const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid, Plots, Printf, LinearAlgebra, JLD
import MPI

norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))

@views inn(A) = A[2:end-1,2:end-1,2:end-1]

macro innH3()   esc(:( @inn(H)*@inn(H)*@inn(H) )) end
macro av_xiH3() esc(:( @av_xi(H)*@av_xi(H)*@av_xi(H) )) end
macro av_yiH3() esc(:( @av_yi(H)*@av_yi(H)*@av_yi(H) )) end
macro av_ziH3() esc(:( @av_zi(H)*@av_zi(H)*@av_zi(H) )) end
macro av_xi_Re_opt() esc(:( π + sqrt(π^2 + (lx/@av_xiH3())^2) )) end
macro av_yi_Re_opt() esc(:( π + sqrt(π^2 + (lx/@av_yiH3())^2) )) end
macro av_zi_Re_opt() esc(:( π + sqrt(π^2 + (lx/@av_ziH3())^2) )) end
macro av_xi_dtauq()  esc(:( dmp*CFLdx*lx/@av_xi_Re_opt() )) end
macro av_yi_dtauq()  esc(:( dmp*CFLdx*lx/@av_yi_Re_opt() )) end
macro av_zi_dtauq()  esc(:( dmp*CFLdx*lx/@av_zi_Re_opt() )) end
macro Re_opt()  esc(:( π + sqrt(π^2 + (lx/@innH3())^2)  )) end
macro dtauH()   esc(:( CFLdx^2/(dmp*CFLdx*lx/@Re_opt()) )) end # dtauH*dtauq = CFL^2*dx^2 -> dt < CFL*dx/Vsound

@parallel function compute_flux!(qHx, qHy, qHz, qHx2, qHy2, qHz2, H, dmp, CFLdx, lx, dx, dy, dz)
    @all(qHx)  = (@all(qHx) - @av_xi_dtauq()*@d_xi(H)/dx)/(1.0 + @av_xi_dtauq()/@av_xiH3())
    @all(qHy)  = (@all(qHy) - @av_yi_dtauq()*@d_yi(H)/dy)/(1.0 + @av_yi_dtauq()/@av_yiH3())
    @all(qHz)  = (@all(qHz) - @av_zi_dtauq()*@d_zi(H)/dz)/(1.0 + @av_zi_dtauq()/@av_ziH3())
    @all(qHx2) = -@av_xiH3()*@d_xi(H)/dx
    @all(qHy2) = -@av_yiH3()*@d_yi(H)/dy
    @all(qHz2) = -@av_ziH3()*@d_zi(H)/dz
    return
end

@parallel function compute_update!(H, Hold, qHx, qHy, qHz, dt, dmp, CFLdx, lx, dx, dy, dz)
    @inn(H) = (@inn(H) + @dtauH()*(@inn(Hold)/dt - (@d_xa(qHx)/dx + @d_ya(qHy)/dy + @d_za(qHz)/dz)))/(1.0 + @dtauH()/dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx2, qHy2, qHz2, dt, dx, dy, dz)
    @all(ResH) = -(@inn(H)-@inn(Hold))/dt - (@d_xa(qHx2)/dx + @d_ya(qHy2)/dy + @d_za(qHz2)/dz)
    return
end

@views function diffusion_3D(; nx=32, ny=32, nz=32, MPI_ini_fin=true, do_viz=false)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    ttot       = 1.0              # total simulation time
    dt         = 0.2              # physical time step
    # Numerics
    tol        = 1e-8             # tolerance
    itMax      = 1e5              # max number of iterations
    nout       = 10               # tol check
    me, dims   = init_global_grid(nx, ny, nz; init_MPI=MPI_ini_fin) # MPI initialisation
    @static if USE_GPU select_device() end    # select one GPU per MPI local rank (if >1 GPU per node)
    b_width    = (8, 4, 4)       # boundary width for comm/comp overlap
    # Derived numerics    
    dx, dy, dz = lx/nx_g(), ly/ny_g(), lz/nz_g() # cell sizes
    dmp        = 4.5
    CFLdx      = 0.5*dx # instead of 0.7 (2D, 1D)
    # Array allocation
    qHx        = @zeros(nx-1,ny-2,nz-2)
    qHy        = @zeros(nx-2,ny-1,nz-2)
    qHz        = @zeros(nx-2,ny-2,nz-1)
    qHx2       = @zeros(nx-1,ny-2,nz-2)
    qHy2       = @zeros(nx-2,ny-1,nz-2)
    qHz2       = @zeros(nx-2,ny-2,nz-1)
    ResH       = @zeros(nx-2,ny-2,nz-2)
    # Initial condition
    H0         = zeros(nx,ny,nz)
    H0         = Data.Array([exp(-(x_g(ix,dx,H0)-0.5*lx+dx/2)*(x_g(ix,dx,H0)-0.5*lx+dx/2) - (y_g(iy,dy,H0)-0.5*ly+dy/2)*(y_g(iy,dy,H0)-0.5*ly+dy/2) - (z_g(iz,dz,H0)-0.5*lz+dz/2)*(z_g(iz,dz,H0)-0.5*lz+dz/2)) for ix=1:size(H0,1), iy=1:size(H0,2), iz=1:size(H0,3)])
    Hold       = @ones(nx,ny,nz).*H0
    H          = @ones(nx,ny,nz).*H0
    len_ResH_g = ((nx-2-2)*dims[1]+2)*((ny-2-2)*dims[2]+2)*((nz-2-2)*dims[3]+2)
    if do_viz
        if (me==0) ENV["GKSwstype"]="nul"; !ispath("../../figures") && mkdir("../../figures") end
        nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
        if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        H_v    = zeros(nx_v, ny_v, nz_v) # global array for visu
        H_inn  = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
        z_sl   = Int(ceil(nz_g()/2))     # Central z-slice
        Xi_g, Yi_g = dx+dx/2:dx:lx-dx-dx/2, dy+dy/2:dy:ly-dy-dy/2 # inner points only
    end
    t = 0.0; it = 0; ittot = 0
    # Physical time loop
    while t<ttot
        iter = 0; err = 2*tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax
            @parallel compute_flux!(qHx, qHy, qHz, qHx2, qHy2, qHz2, H, dmp, CFLdx, lx, dx, dy, dz)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_update!(H, Hold, qHx, qHy, qHz, dt, dmp, CFLdx, lx, dx, dy, dz)
                update_halo!(H)
            end
            iter += 1
            if iter % nout == 0
                @parallel check_res!(ResH, H, Hold, qHx2, qHy2, qHz2, dt, dx, dy, dz)
                err = norm_g(ResH)/len_ResH_g
            end
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end
    if (me==0) @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx_g(), ittot) end
    # Visualise
    if do_viz 
        H_inn .= inn(H); gather!(H_inn, H_v)
        if me==0
            heatmap(Xi_g, Yi_g, H_v[:,:,z_sl]', dpi=150, aspect_ratio=1, framestyle=:box, xlims=(Xi_g[1],Xi_g[end]), ylims=(Yi_g[1],Yi_g[end]), xlabel="lx", ylabel="ly", c=:hot, clims=(0,1), title="nonlinear diffusion (nt=$it, iters=$ittot)")
            savefig("../../figures/diff3Dnonlin2_$(nx_g()).png")
        end
    end
    nxg, nyg, nzg = nx_g(), ny_g(), nz_g()
    finalize_global_grid(; finalize_MPI=MPI_ini_fin)
    return nxg, nyg, nzg, ittot, me
end

# diffusion_3D(; nx=128, ny=128, nz=128, do_viz=false)

@views function runtests_3D(name; do_save=false)

    resol = 16 * 2 .^ (1:5)

    out = zeros(4, length(resol))
    me  = 0
    
    MPI.Init()
    
    for i = 1:length(resol)

        res = resol[i]

        nxx, nyy, nzz, iter, me = diffusion_3D(; nx=res, ny=res, nz=res, MPI_ini_fin=false)

        out[1,i] = nxx
        out[2,i] = nyy
        out[3,i] = nzz
        out[4,i] = iter
    end

    if do_save && me==0
        !ispath("../../output") && mkdir("../../output")
        save("../../output/out_$(name).jld", "out", out)
    end

    MPI.Finalize()
end

runtests_3D("diff_3D_nonlin2"; do_save=true)

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

@parallel function compute_dtau!(dtau, D, dt, dx, dy, dz)
    # @all(dtau) = 1.0./(1.0./(min(dx,dy,dz)^2 ./@inn(D)/6.1) .+ 1.0/dt)
    @all(dtau) = 1.0./(1.0./(min(dx,dy,dz)^2 ./@maxloc(D)/6.1) .+ 1.0/dt)
    return
end

@parallel function compute_flux!(qHx, qHy, qHz, H, D, dx, dy, dz)
    @all(qHx) = -@av_xi(D)*@d_xi(H)/dx
    @all(qHy) = -@av_yi(D)*@d_yi(H)/dy
    @all(qHz) = -@av_zi(D)*@d_zi(H)/dz
    return
end

@parallel function compute_rate!(ResH, dHdt, H, Hold, qHx, qHy, qHz, dt, damp, dx, dy, dz)
    @all(ResH) = -(@inn(H) - @inn(Hold))/dt - (@d_xa(qHx)/dx + @d_ya(qHy)/dy + @d_za(qHz)/dz)
    @all(dHdt) = @all(ResH) + damp*@all(dHdt)
    return
end

@parallel function compute_update!(H, dHdt, dtau)
    @inn(H) = @inn(H) + @all(dtau)*@all(dHdt)
    return
end

@views function diffusion_3D(; nx=32, ny=32, nz=32, MPI_ini_fin=true, do_viz=false)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D1         = 1.0              # diffusion coefficient
    D2         = 1e-4             # diffusion coefficient
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
    damp       = 1-18/nx_g()      # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    dx, dy, dz = lx/nx_g(), ly/ny_g(), lz/nz_g()           # cell sizes
    # Array allocation
    qHx        = @zeros(nx-1,ny-2,nz-2)
    qHy        = @zeros(nx-2,ny-1,nz-2)
    qHz        = @zeros(nx-2,ny-2,nz-1)
    dHdt       = @zeros(nx-2,ny-2,nz-2)
    ResH       = @zeros(nx-2,ny-2,nz-2)
    dtau       = @zeros(nx-2,ny-2,nz-2)
    # Initial condition
    H0         =   zeros(nx,ny,nz)
    D          = D1*ones(nx,ny,nz)
    Tmp        = [x_g(ix,dx,H0) for ix=1:size(H0,1), iy=1:size(H0,2), iz=1:size(H0,3)]
    D[Tmp.<lx/2.2] .= D2
    Tmp        = [y_g(iy,dy,H0) for ix=1:size(H0,1), iy=1:size(H0,2), iz=1:size(H0,3)]
    D[Tmp.<ly/2.2] .= D2
    D          = Data.Array(D)
    H0         = Data.Array([exp(-(x_g(ix,dx,H0)-0.5*lx+dx/2)*(x_g(ix,dx,H0)-0.5*lx+dx/2) - (y_g(iy,dy,H0)-0.5*ly+dy/2)*(y_g(iy,dy,H0)-0.5*ly+dy/2) - (z_g(iz,dz,H0)-0.5*lz+dz/2)*(z_g(iz,dz,H0)-0.5*lz+dz/2)) for ix=1:size(H0,1), iy=1:size(H0,2), iz=1:size(H0,3)])
    Hold       = @ones(nx,ny,nz).*H0
    H          = @ones(nx,ny,nz).*H0
    len_ResH_g = ((nx-2-2)*dims[1]+2)*((ny-2-2)*dims[2]+2)*((nz-2-2)*dims[3]+2)
    @parallel compute_dtau!(dtau, D, dt, dx, dy, dz)
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
            @parallel compute_flux!(qHx, qHy, qHz, H, D, dx, dy, dz)
            @parallel compute_rate!(ResH, dHdt, H, Hold, qHx, qHy, qHz, dt, damp, dx, dy, dz)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_update!(H, dHdt, dtau)
                update_halo!(H)
            end
            iter += 1; if (iter % nout == 0)  err = norm_g(ResH)/len_ResH_g  end
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
            heatmap(Xi_g, Yi_g, H_v[:,:,z_sl]', dpi=150, aspect_ratio=1, framestyle=:box, xlims=(Xi_g[1],Xi_g[end]), ylims=(Yi_g[1],Yi_g[end]), xlabel="lx", ylabel="ly", c=:hot, clims=(0,1), title="linear step diffusion (nt=$it, iters=$ittot)")
            savefig("../../figures/diff3Dlinstep_$(nx_g()).png")
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
    return
end

runtests_3D("diff_3D_linstep"; do_save=true)

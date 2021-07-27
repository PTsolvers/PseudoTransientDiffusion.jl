const USE_GPU = parse(Bool, ENV["USE_GPU"])
const do_viz  = parse(Bool, ENV["DO_VIZ"])
const do_save = parse(Bool, ENV["DO_SAVE"])
const do_save_viz = parse(Bool, ENV["DO_SAVE_VIZ"])
const nx = parse(Int, ENV["NX"])
const ny = parse(Int, ENV["NY"])
const nz = parse(Int, ENV["NZ"])
###
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid, Plots, Printf, LinearAlgebra, MAT
import MPI

norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))

@views inn(A) = A[2:end-1,2:end-1,2:end-1]

@parallel function compute_Re!(Re, D, max_lxyz, dt)
    @inn(Re) = π + sqrt(π^2 + (max_lxyz^2 / @maxloc(D) / dt))
    return
end

@parallel function compute_iter_params!(τr_dt, dt_ρ, Re, D, Vpdt, max_lxyz)
    @all(τr_dt) = max_lxyz / Vpdt / @all(Re)
    @all(dt_ρ)  = Vpdt * max_lxyz / @maxloc(D) / @inn(Re)
    return
end

@parallel function compute_flux!(qHx, qHy, qHz, qHx2, qHy2, qHz2, H, D, τr_dt, dx, dy, dz)
    @all(qHx)  = (@all(qHx) * @av_xi(τr_dt) - @av_xi(D) * @d_xi(H) / dx) / (1.0 + @av_xi(τr_dt))
    @all(qHy)  = (@all(qHy) * @av_yi(τr_dt) - @av_yi(D) * @d_yi(H) / dy) / (1.0 + @av_yi(τr_dt))
    @all(qHz)  = (@all(qHz) * @av_zi(τr_dt) - @av_zi(D) * @d_zi(H) / dz) / (1.0 + @av_zi(τr_dt))
    @all(qHx2) = -@av_xi(D) * @d_xi(H) / dx
    @all(qHy2) = -@av_yi(D) * @d_yi(H) / dy
    @all(qHz2) = -@av_zi(D) * @d_zi(H) / dz
    return
end

@parallel function compute_update!(H, Hold, qHx, qHy, qHz, dt_ρ, dt, dx, dy, dz)
    @inn(H) = (@inn(H) + @all(dt_ρ) * (@inn(Hold) / dt - (@d_xa(qHx) / dx + @d_ya(qHy) / dy + @d_za(qHz)/dz))) / (1.0 + @all(dt_ρ) / dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx2, qHy2, qHz2, dt, dx, dy, dz)
    @all(ResH) = -(@inn(H) - @inn(Hold)) / dt - (@d_xa(qHx2) / dx + @d_ya(qHy2) / dy + @d_za(qHz2) / dz)
    return
end

@parallel_indices (iy,iz) function bc_x!(A)
    A[1  , iy, iz] = A[2    , iy, iz]
    A[end, iy, iz] = A[end-1, iy, iz]
    return
end

@parallel_indices (ix,iz) function bc_y!(A)
    A[ix, 1  , iz] = A[ix, 2    , iz]
    A[ix, end, iz] = A[ix, end-1, iz]
    return
end

@parallel_indices (ix,iy) function bc_z!(A)
    A[ix, iy, 1  ] = A[ix, iy, 2    ]
    A[ix, iy, end] = A[ix, iy, end-1]
    return
end

@views function diffusion_3D()
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
    CFL        = 1/sqrt(3)        # CFL number
    me, dims   = init_global_grid(nx, ny, nz) # MPI initialisation
    @static if USE_GPU select_device() end    # select one GPU per MPI local rank (if >1 GPU per node)
    b_width    = (8, 4, 4)       # boundary width for comm/comp overlap
    # Derived numerics    
    dx, dy, dz = lx/nx_g(), ly/ny_g(), lz/nz_g() # cell sizes
    Vpdt       = CFL * min(dx, dy, dz)
    max_lxyz   = max(lx, ly, lz)
    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    # Array allocation
    qHx        = @zeros(nx-1,ny-2,nz-2)
    qHy        = @zeros(nx-2,ny-1,nz-2)
    qHz        = @zeros(nx-2,ny-2,nz-1)
    qHx2       = @zeros(nx-1,ny-2,nz-2)
    qHy2       = @zeros(nx-2,ny-1,nz-2)
    qHz2       = @zeros(nx-2,ny-2,nz-1)
    ResH       = @zeros(nx-2,ny-2,nz-2)
    Re         = @zeros(nx  ,ny  ,nz  )
    τr_dt      = @zeros(nx  ,ny  ,nz  )
    dt_ρ       = @zeros(nx-2,ny-2,nz-2)
    # Initial condition
    H0         =   zeros(nx,ny,nz)
    D          = D1*ones(nx,ny,nz)
    Tmp        = [x_g(ix,dx,H0) for ix=1:size(H0,1), iy=1:size(H0,2), iz=1:size(H0,3)]
    D[Tmp.<lx/2.2] .= D2
    Tmp        = [y_g(iy,dy,H0) for ix=1:size(H0,1), iy=1:size(H0,2), iz=1:size(H0,3)]
    D[Tmp.<ly/2.2] .= D2
    D          = Data.Array(D)
    H0         = Data.Array([exp(-(x_g(ix,dx,H0)-0.5*lx+dx/2)*(x_g(ix,dx,H0)-0.5*lx+dx/2) - (y_g(iy,dy,H0)-0.5*ly+dy/2)*(y_g(iy,dy,H0)-0.5*ly+dy/2) - (z_g(iz,dz,H0)-0.5*lz+dz/2)*(z_g(iz,dz,H0)-0.5*lz+dz/2)) for ix=1:size(H0,1), iy=1:size(H0,2), iz=1:size(H0,3)])
    Hold       = @ones(nx,ny,nz) .* H0
    H          = @ones(nx,ny,nz) .* H0
    @parallel compute_Re!(Re, D, max_lxyz, dt)
    @parallel (1:size(Re,2),1:size(Re,2)) bc_z!(Re)
    @parallel (1:size(Re,1),1:size(Re,3)) bc_y!(Re)
    @parallel (1:size(Re,2),1:size(Re,3)) bc_x!(Re)
    @parallel compute_iter_params!(τr_dt, dt_ρ, Re, D, Vpdt, max_lxyz)
    len_ResH_g = ((nx-2-2)*dims[1]+2)*((ny-2-2)*dims[2]+2)*((nz-2-2)*dims[3]+2)
    if do_viz || do_save_viz
        if (me==0) ENV["GKSwstype"]="nul"; if do_viz !ispath("../../figures") && mkdir("../../figures") end; end
        nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
        if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        H_v    = zeros(nx_v, ny_v, nz_v) # global array for visu
        H_inn  = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
        z_sl   = Int(ceil(nz_g()/2))     # Central z-slice
        Xi_g, Yi_g = dx+dx/2:dx:lx-dx-dx/2, dy+dy/2:dy:ly-dy-dy/2 # inner points only
    end
    t = 0.0; it = 0; ittot = 0; nt = Int(ceil(ttot/dt))
    # Physical time loop
    while it<nt
        iter = 0; err = 2 * tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax
            @parallel compute_flux!(qHx, qHy, qHz, qHx2, qHy2, qHz2, H, D, τr_dt, dx, dy, dz)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_update!(H, Hold, qHx, qHy, qHz, dt_ρ, dt, dx, dy, dz)
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
    if do_viz || do_save_viz
        H_inn .= inn(H); gather!(H_inn, H_v)
        if me==0 && do_viz
            heatmap(Xi_g, Yi_g, H_v[:,:,z_sl]', dpi=150, aspect_ratio=1, framestyle=:box, xlims=(Xi_g[1],Xi_g[end]), ylims=(Yi_g[1],Yi_g[end]), xlabel="lx", ylabel="ly", c=:hot, clims=(0,1), title="linear step diffusion (nt=$it, iters=$ittot)")
            savefig("../../figures/diff_3D_linstep3_$(nx_g()).png")
        end
    end
    if me==0 && do_save
        !ispath("../../output") && mkdir("../../output")
        open("../../output/out_diff_3D_linstep3.txt","a") do io
            println(io, "$(nx_g()) $(ny_g()) $(nz_g()) $(ittot) $(nt)")
        end
    end
    if me==0 && do_save_viz
        !ispath("../../out_visu") && mkdir("../../out_visu")
        matwrite("../../out_visu/diff_3D_linstep3.mat", Dict("H_3D"=> Array(H_v), "xc_3D"=> Array(xc), "yc_3D"=> Array(yc), "zc_3D"=> Array(zc)); compress = true)
    end
    finalize_global_grid()
    return
end

diffusion_3D()

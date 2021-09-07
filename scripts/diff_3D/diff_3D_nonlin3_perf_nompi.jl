const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const do_viz  = haskey(ENV, "DO_VIZ")  ? parse(Bool, ENV["DO_VIZ"])  : false
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : false
const do_save_viz = haskey(ENV, "DO_SAVE_VIZ") ? parse(Bool, ENV["DO_SAVE_VIZ"]) : false
const nx = haskey(ENV, "NX") ? parse(Int, ENV["NX"]) : 64
const ny = haskey(ENV, "NY") ? parse(Int, ENV["NY"]) : 64
const nz = haskey(ENV, "NZ") ? parse(Int, ENV["NZ"]) : 64
###
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using Plots, Printf, LinearAlgebra, MAT

macro innH3(ix,iy,iz)       esc(:( H[$ix+1,$iy+1,$iz+1]*H[$ix+1,$iy+1,$iz+1]*H[$ix+1,$iy+1,$iz+1] )) end
macro av_xi_H3(ix,iy,iz)    esc(:( 0.5*(H[$ix,$iy+1,$iz+1]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix,$iy+1,$iz+1]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix,$iy+1,$iz+1]+H[$ix+1,$iy+1,$iz+1]) )) end
macro av_yi_H3(ix,iy,iz)    esc(:( 0.5*(H[$ix+1,$iy,$iz+1]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix+1,$iy,$iz+1]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix+1,$iy,$iz+1]+H[$ix+1,$iy+1,$iz+1]) )) end
macro av_zi_H3(ix,iy,iz)    esc(:( 0.5*(H[$ix+1,$iy+1,$iz]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix+1,$iy+1,$iz]+H[$ix+1,$iy+1,$iz+1]) * 0.5*(H[$ix+1,$iy+1,$iz]+H[$ix+1,$iy+1,$iz+1]) )) end
macro av_xi_Re(ix,iy,iz)    esc(:( π + sqrt(π*π + max_lxyz2 / @av_xi_H3($ix,$iy,$iz) * _dt) )) end
macro av_yi_Re(ix,iy,iz)    esc(:( π + sqrt(π*π + max_lxyz2 / @av_yi_H3($ix,$iy,$iz) * _dt) )) end
macro av_zi_Re(ix,iy,iz)    esc(:( π + sqrt(π*π + max_lxyz2 / @av_zi_H3($ix,$iy,$iz) * _dt) )) end
macro Re(ix,iy,iz)          esc(:( π + sqrt(π*π + max_lxyz2 / @innH3($ix,$iy,$iz)    * _dt) )) end
macro av_xi_τr_dt(ix,iy,iz) esc(:( max_lxyz / Vpdt / @av_xi_Re($ix,$iy,$iz) * Resc )) end
macro av_yi_τr_dt(ix,iy,iz) esc(:( max_lxyz / Vpdt / @av_yi_Re($ix,$iy,$iz) * Resc )) end
macro av_zi_τr_dt(ix,iy,iz) esc(:( max_lxyz / Vpdt / @av_zi_Re($ix,$iy,$iz) * Resc )) end
macro dt_ρ(ix,iy,iz)        esc(:( Vpdt * max_lxyz / @innH3($ix,$iy,$iz) / @Re($ix,$iy,$iz) * Resc )) end

@parallel_indices (ix,iy,iz) function compute_flux!(qHx, qHy, qHz, H, Vpdt, Resc, _dt, max_lxyz, max_lxyz2, _dx, _dy, _dz)
    if (ix<=size(qHx,1) && iy<=size(qHx,2) && iz<=size(qHx,3)) qHx[ix,iy,iz]  = (qHx[ix,iy,iz] * @av_xi_τr_dt(ix,iy,iz) - @av_xi_H3(ix,iy,iz) * _dx * (H[ix+1,iy+1,iz+1] - H[ix,iy+1,iz+1]) ) / (1.0 + @av_xi_τr_dt(ix,iy,iz))  end
    if (ix<=size(qHy,1) && iy<=size(qHy,2) && iz<=size(qHy,3)) qHy[ix,iy,iz]  = (qHy[ix,iy,iz] * @av_yi_τr_dt(ix,iy,iz) - @av_yi_H3(ix,iy,iz) * _dy * (H[ix+1,iy+1,iz+1] - H[ix+1,iy,iz+1]) ) / (1.0 + @av_yi_τr_dt(ix,iy,iz))  end
    if (ix<=size(qHz,1) && iy<=size(qHz,2) && iz<=size(qHz,3)) qHz[ix,iy,iz]  = (qHz[ix,iy,iz] * @av_zi_τr_dt(ix,iy,iz) - @av_zi_H3(ix,iy,iz) * _dz * (H[ix+1,iy+1,iz+1] - H[ix+1,iy+1,iz]) ) / (1.0 + @av_zi_τr_dt(ix,iy,iz))  end
    return
end

@parallel_indices (ix,iy,iz) function compute_update!(H, Hold, qHx, qHy, qHz, Vpdt, Resc, _dt, max_lxyz, max_lxyz2, _dx, _dy, _dz, size_innH_1, size_innH_2, size_innH_3)
    if (ix<=size_innH_1 && iy<=size_innH_2 && iz<=size_innH_3)  H[ix+1,iy+1,iz+1] = (H[ix+1,iy+1,iz+1] + @dt_ρ(ix,iy,iz) * (_dt * Hold[ix+1,iy+1,iz+1] - (_dx * (qHx[ix+1,iy,iz] - qHx[ix,iy,iz]) + _dy * (qHy[ix,iy+1,iz] - qHy[ix,iy,iz])  + _dz * (qHz[ix,iy,iz+1] - qHz[ix,iy,iz])) )) / (1.0 + _dt * @dt_ρ(ix,iy,iz))  end
    return
end

@parallel_indices (ix,iy,iz) function compute_flux_res!(qHx2, qHy2, qHz2, H, _dx, _dy, _dz)
    if (ix<=size(qHx2,1) && iy<=size(qHx2,2) && iz<=size(qHx2,3))  qHx2[ix,iy,iz] = -@av_xi_H3(ix,iy,iz) * _dx * (H[ix+1,iy+1,iz+1] - H[ix,iy+1,iz+1])  end
    if (ix<=size(qHy2,1) && iy<=size(qHy2,2) && iz<=size(qHy2,3))  qHy2[ix,iy,iz] = -@av_yi_H3(ix,iy,iz) * _dy * (H[ix+1,iy+1,iz+1] - H[ix+1,iy,iz+1])  end
    if (ix<=size(qHz2,1) && iy<=size(qHz2,2) && iz<=size(qHz2,3))  qHz2[ix,iy,iz] = -@av_zi_H3(ix,iy,iz) * _dz * (H[ix+1,iy+1,iz+1] - H[ix+1,iy+1,iz])  end
    return
end

@parallel_indices (ix,iy,iz) function check_res!(ResH, H, Hold, qHx2, qHy2, qHz2, _dt, _dx, _dy, _dz)
    if (ix<=size(ResH,1) && iy<=size(ResH,2) && iz<=size(ResH,3))  ResH[ix,iy,iz] = -_dt * (H[ix+1,iy+1,iz+1] - Hold[ix+1,iy+1,iz+1]) - (_dx * (qHx[ix+1,iy,iz] - qHx[ix,iy,iz]) + _dy * (qHy[ix,iy+1,iz] - qHy[ix,iy,iz])  + _dz * (qHz[ix,iy,iz+1] - qHz[ix,iy,iz]))  end
    return
end

@parallel_indices (ix,iy,iz) function assign!(Hold, H)
    if (ix<=size(H,1) && iy<=size(H,2) && iz<=size(H,3))  Hold[ix,iy,iz] = H[ix,iy,iz]  end
    return
end

@views function diffusion_3D()
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    ttot       = 0.4              # total simulation time
    dt         = 0.2              # physical time step
    # Numerics
    tol        = 1e-8             # tolerance
    itMax      = 1e3              # max number of iterations
    nout       = 2000             # tol check
    CFL        = 1 / sqrt(3)      # CFL number
    Resc       = 1 / 1.2          # iteration parameter scaling
    # Derived numerics    
    dx, dy, dz = lx/nx, ly/ny, lz/nz # cell sizes
    Vpdt       = CFL * min(dx, dy, dz)
    max_lxyz   = max(lx, ly, lz)
    max_lxyz2  = max_lxyz^2
    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    _dx, _dy, _dz, _dt = 1.0/dx, 1.0/dy, 1.0/dz, 1.0/dt
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
    H0         = Data.Array([exp(-(xc[ix]-0.5*lx)^2 - (yc[iy]-0.5*ly)^2 - (zc[iz]-0.5*lz)^2) for ix=1:size(H0,1), iy=1:size(H0,2), iz=1:size(H0,3)])
    Hold       = @ones(nx,ny,nz) .* H0
    H          = @ones(nx,ny,nz) .* H0
    size_innH_1, size_innH_2, size_innH_3 = size(H,1)-2, size(H,2)-2, size(H,3)-2
    t = 0.0; it = 0; ittot = 0; nt = Int(ceil(ttot/dt)); t_tic = 0.0; niter = 0
    # Physical time loop
    while it<nt
        iter = 0; err = 2*tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax
            if (it==1 && iter==0) t_tic = Base.time(); niter = 0 end
            @parallel compute_flux!(qHx, qHy, qHz, H, Vpdt, Resc, _dt, max_lxyz, max_lxyz2, _dx, _dy, _dz)
            @parallel compute_update!(H, Hold, qHx, qHy, qHz, Vpdt, Resc, _dt, max_lxyz, max_lxyz2, _dx, _dy, _dz, size_innH_1, size_innH_2, size_innH_3)
            iter += 1;  niter += 1
            if iter % nout == 0
                @parallel compute_flux_res!(qHx2, qHy2, qHz2, H, _dx, _dy, _dz)
                @parallel check_res!(ResH, H, Hold, qHx2, qHy2, qHz2, _dt, _dx, _dy, _dz)
                err = norm(ResH)/length(ResH)
            end
        end
        ittot += iter; it += 1; t += dt
        @parallel assign!(Hold, H)
        if isnan(err) error("NaN") end
    end
    t_toc = Base.time() - t_tic
    A_eff = (2*4+2)/1e9*nx*ny*nz*sizeof(Data.Number) # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                              # Execution time per iteration [s]
    T_eff = A_eff/t_it                               # Effective memory throughput [GB/s]
    @printf("PERF: Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter)
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz
        !ispath("../../figures") && mkdir("../../figures")
        heatmap(xc, yc, Array(H)[:,:,Int(ceil(nz/2))]', dpi=150, aspect_ratio=1, framestyle=:box, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), xlabel="lx", ylabel="ly", c=:viridis, clims=(0,1), title="nonlinear diffusion (nt=$it, iters=$ittot)")
        savefig("../../figures/diff_3D_nonlin3_perf_$(nx).png")
    end
    if do_save
        !ispath("../../output") && mkdir("../../output")
        open("../../output/out_diff_3D_nonlin3_perf.txt","a") do io
            println(io, "$(nx) $(ny) $(nz) $(ittot) $(nt)")
        end
    end
    if do_save_viz
        !ispath("../../out_visu") && mkdir("../../out_visu")
        matwrite("../../out_visu/diff_3D_nonlin3_perf.mat", Dict("H_3D"=> Array(H), "xc_3D"=> Array(xc), "yc_3D"=> Array(yc), "zc_3D"=> Array(zc)); compress = true)
    end
    return
end

diffusion_3D()

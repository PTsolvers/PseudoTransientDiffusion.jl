const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const do_viz  = haskey(ENV, "DO_VIZ")  ? parse(Bool, ENV["DO_VIZ"])  : false
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : false
const do_save_viz = haskey(ENV, "DO_SAVE_VIZ") ? parse(Bool, ENV["DO_SAVE_VIZ"]) : false
const nx = haskey(ENV, "NX") ? parse(Int, ENV["NX"]) : 256
const ny = haskey(ENV, "NY") ? parse(Int, ENV["NY"]) : 256
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    # CUDA.device!(6) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, LinearAlgebra, MAT

macro innH3(ix,iy)       esc(:( H[$ix+1,$iy+1] * H[$ix+1,$iy+1] * H[$ix+1,$iy+1] )) end
macro av_xi_H3(ix,iy)    esc(:( 0.5*(H[$ix,$iy+1]+H[$ix+1,$iy+1]) * 0.5*(H[$ix,$iy+1]+H[$ix+1,$iy+1]) * 0.5*(H[$ix,$iy+1]+H[$ix+1,$iy+1]) )) end
macro av_yi_H3(ix,iy)    esc(:( 0.5*(H[$ix+1,$iy]+H[$ix+1,$iy+1]) * 0.5*(H[$ix+1,$iy]+H[$ix+1,$iy+1]) * 0.5*(H[$ix+1,$iy]+H[$ix+1,$iy+1]) )) end
macro av_xi_Re(ix,iy)    esc(:( π + sqrt(π*π + max_lxy2 / @av_xi_H3($ix,$iy) * _dt) )) end
macro av_yi_Re(ix,iy)    esc(:( π + sqrt(π*π + max_lxy2 / @av_yi_H3($ix,$iy) * _dt) )) end
macro Re(ix,iy)          esc(:( π + sqrt(π*π + max_lxy2 / @innH3($ix,$iy)    * _dt) )) end
macro av_xi_τr_dt(ix,iy) esc(:( max_lxy / Vpdt / @av_xi_Re($ix,$iy) * Resc )) end
macro av_yi_τr_dt(ix,iy) esc(:( max_lxy / Vpdt / @av_yi_Re($ix,$iy) * Resc )) end
macro dt_ρ(ix,iy)        esc(:( Vpdt * max_lxy / @innH3($ix,$iy) / @Re($ix,$iy) * Resc )) end

@parallel_indices (ix,iy) function compute_flux!(qHx, qHy, H, Vpdt, Resc, _dt, max_lxy, max_lxy2, _dx, _dy)
    if (ix<=size(qHx,1) && iy<=size(qHx,2))  qHx[ix,iy]  = (qHx[ix,iy] * @av_xi_τr_dt(ix,iy) - @av_xi_H3(ix,iy) * _dx * (H[ix+1,iy+1] - H[ix,iy+1]) ) / (1.0 + @av_xi_τr_dt(ix,iy))  end
    if (ix<=size(qHy,1) && iy<=size(qHy,2))  qHy[ix,iy]  = (qHy[ix,iy] * @av_yi_τr_dt(ix,iy) - @av_yi_H3(ix,iy) * _dy * (H[ix+1,iy+1] - H[ix+1,iy]) ) / (1.0 + @av_yi_τr_dt(ix,iy))  end
    return
end

@parallel_indices (ix,iy) function compute_update!(H, Hold, qHx, qHy, Vpdt, Resc, _dt, max_lxy, max_lxy2, _dx, _dy, size_innH_1, size_innH_2)
    if (ix<=size_innH_1 && iy<=size_innH_2)  H[ix+1,iy+1] = (H[ix+1,iy+1] + @dt_ρ(ix,iy) * (_dt * Hold[ix+1,iy+1] - (_dx * (qHx[ix+1,iy] - qHx[ix,iy]) + _dy * (qHy[ix,iy+1] - qHy[ix,iy])) )) / (1.0 + _dt * @dt_ρ(ix,iy))  end
    return
end

@parallel_indices (ix,iy) function compute_flux_res!(qHx2, qHy2, H, _dx, _dy)
    if (ix<=size(qHx2,1) && iy<=size(qHx2,2))  qHx2[ix,iy] = -@av_xi_H3(ix,iy) * _dx * (H[ix+1,iy+1] - H[ix,iy+1])  end
    if (ix<=size(qHy2,1) && iy<=size(qHy2,2))  qHy2[ix,iy] = -@av_yi_H3(ix,iy) * _dy * (H[ix+1,iy+1] - H[ix+1,iy])  end
    return
end

@parallel_indices (ix,iy) function check_res!(ResH, H, Hold, qHx2, qHy2, _dt, _dx, _dy)
    if (ix<=size(ResH,1) && iy<=size(ResH,2))  ResH[ix,iy] = -_dt * (H[ix+1,iy+1] - Hold[ix+1,iy+1]) - (_dx * (qHx2[ix+1,iy] - qHx2[ix,iy]) + _dy * (qHy2[ix,iy+1] - qHy2[ix,iy]))  end
    return
end

@parallel_indices (ix,iy) function assign!(Hold, H)
    if (ix<=size(H,1) && iy<=size(H,2))  Hold[ix,iy] = H[ix,iy]  end
    return
end

@views function diffusion_2D()
    # Physics
    lx, ly  = 10.0, 10.0    # domain size
    ttot    = 0.4           # total simulation time
    dt      = 0.2           # physical time step
    # Numerics
    # nx, ny  = 2*256, 2*256  # numerical grid resolution
    tol     = 1e-8          # tolerance
    itMax   = 1e2#1e5           # max number of iterations
    nout    = 2000            # tol check
    CFL     = 1 / sqrt(2)   # CFL number
    Resc    = 1 / 1.2       # iteration parameter scaling
    # Derived numerics
    dx, dy  = lx / nx, ly / ny  # grid size   
    Vpdt    = CFL * min(dx, dy)
    max_lxy = max(lx, ly)
    max_lxy2= max_lxy^2
    xc, yc  = LinRange(-lx / 2, lx / 2, nx), LinRange(-ly / 2, ly / 2, ny)
    _dx, _dy, _dt = 1.0/dx, 1.0/dy, 1.0/dt
    # Array allocation
    qHx     = @zeros(nx-1, ny-2)
    qHy     = @zeros(nx-2, ny-1)
    qHx2    = @zeros(nx-1, ny-2)
    qHy2    = @zeros(nx-2, ny-1)
    ResH    = @zeros(nx-2, ny-2)
    # Initial condition
    H0      = Data.Array(exp.(-xc.^2 .- (yc').^2))
    Hold    = @ones(nx,ny) .* H0
    H       = @ones(nx,ny) .* H0
    size_innH_1, size_innH_2 = size(H,1)-2, size(H,2)-2
    t = 0.0; it = 0; ittot = 0; nt = Int(ceil(ttot/dt)); t_tic = 0.0; niter = 0
    # Physical time loop
    while it < nt
        iter = 0; err = 2 * tol
        # Pseudo-transient iteration
        while err > tol && iter < itMax
            if (it==1 && iter==0) t_tic = Base.time(); niter = 0 end
            @parallel compute_flux!(qHx, qHy, H, Vpdt, Resc, _dt, max_lxy, max_lxy2, _dx, _dy)
            @parallel compute_update!(H, Hold, qHx, qHy, Vpdt, Resc, _dt, max_lxy, max_lxy2, _dx, _dy, size_innH_1, size_innH_2)
            iter += 1;  niter += 1
            if iter % nout == 0
                @parallel compute_flux_res!(qHx2, qHy2, H, _dx, _dy)
                @parallel check_res!(ResH, H, Hold, qHx2, qHy2, _dt, _dx, _dy)
                err = norm(ResH) / length(ResH)
                if isnan(err) error("NaN") end
            end
        end
        ittot += iter; it += 1; t += dt
        @parallel assign!(Hold, H)
    end
    t_toc = Base.time() - t_tic
    A_eff = (2*3+2)/1e9*nx*ny*sizeof(Data.Number) # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                           # Execution time per iteration [s]
    T_eff = A_eff/t_it                            # Effective memory throughput [GB/s]
    @printf("PERF: Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter)
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz display(heatmap(xc, yc, Array(H'), aspect_ratio=1, framestyle=:box, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), xlabel="lx", ylabel="ly", c=:viridis, clims=(0, 1), title="linear diffusion (nt=$it, iters=$ittot)")) end
    if do_save
        !ispath("../output") && mkdir("../output")
        open("../output/out_diff_2D_nonlin3_perf.txt","a") do io
            println(io, "$(nx) $(ny) $(ittot) $(t_toc) $(A_eff) $(t_it) $(T_eff)")
        end
    end
    if do_save_viz
        !ispath("../out_visu") && mkdir("../out_visu")
        matwrite("../out_visu/diff_2D_nonlin3_perf.mat", Dict("H_2D"=> Array(H), "xc_2D"=> Array(xc), "yc_2D"=> Array(yc)); compress = true)
    end
    return
end

diffusion_2D()

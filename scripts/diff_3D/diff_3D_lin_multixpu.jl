const use_return  = haskey(ENV, "USE_RETURN" ) ? parse(Bool, ENV["USE_RETURN"] ) : false
const USE_GPU     = haskey(ENV, "USE_GPU"    ) ? parse(Bool, ENV["USE_GPU"]    ) : false
const do_viz      = haskey(ENV, "DO_VIZ"     ) ? parse(Bool, ENV["DO_VIZ"]     ) : true
const do_save     = haskey(ENV, "DO_SAVE"    ) ? parse(Bool, ENV["DO_SAVE"]    ) : false
const do_save_viz = haskey(ENV, "DO_SAVE_VIZ") ? parse(Bool, ENV["DO_SAVE_VIZ"]) : false
const nx          = haskey(ENV, "NX"         ) ? parse(Int , ENV["NX"]         ) : 64
const ny          = haskey(ENV, "NY"         ) ? parse(Int , ENV["NY"]         ) : 64
const nz          = haskey(ENV, "NZ"         ) ? parse(Int , ENV["NZ"]         ) : 64
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

@parallel function compute_flux!(qHx, qHy, qHz, qHx2, qHy2, qHz2, H, D, θr_dτ, dx, dy, dz)
    @all(qHx)  = (@all(qHx) * θr_dτ - D * @d_xi(H) / dx) / (1.0 + θr_dτ)
    @all(qHy)  = (@all(qHy) * θr_dτ - D * @d_yi(H) / dy) / (1.0 + θr_dτ)
    @all(qHz)  = (@all(qHz) * θr_dτ - D * @d_zi(H) / dz) / (1.0 + θr_dτ)
    @all(qHx2) = -D * @d_xi(H) / dx
    @all(qHy2) = -D * @d_yi(H) / dy
    @all(qHz2) = -D * @d_zi(H) / dz
    return
end

@parallel function compute_update!(H, Hold, qHx, qHy, qHz, dτ_ρ, dt, dx, dy, dz)
    @inn(H) = (@inn(H) +  dτ_ρ * (@inn(Hold) / dt - (@d_xa(qHx) / dx + @d_ya(qHy) / dy  + @d_za(qHz) / dz))) / (1.0 + dτ_ρ / dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx2, qHy2, qHz2, dt, dx, dy, dz)
    @all(ResH) = -(@inn(H)-@inn(Hold))/dt - (@d_xa(qHx2)/dx + @d_ya(qHy2)/dy + @d_za(qHz2)/dz)
    return
end

@views function diffusion_3D_()
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D          = 1.0              # diffusion coefficient
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
    Vpdτ       = CFL * min(dx, dy, dz)
    Re         = π + sqrt(π^2 + (max(lx, ly, lz)^2 / D / dt)) # Numerical Reynolds number
    θr_dτ      = max(lx, ly, lz) / Vpdτ / Re
    dτ_ρ       = Vpdτ * max(lx, ly, lz) / D / Re
    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    # Array allocation
    qHx        = @zeros(nx-1,ny-2,nz-2)
    qHy        = @zeros(nx-2,ny-1,nz-2)
    qHz        = @zeros(nx-2,ny-2,nz-1)
    qHx2       = @zeros(nx-1,ny-2,nz-2)
    qHy2       = @zeros(nx-2,ny-1,nz-2)
    qHz2       = @zeros(nx-2,ny-2,nz-1)
    ResH       = @zeros(nx-2,ny-2,nz-2)
    # Initial condition
    H0         = zeros(nx, ny, nz)
    H0         = Data.Array([exp(-(x_g(ix,dx,H0)-0.5*lx+dx/2)*(x_g(ix,dx,H0)-0.5*lx+dx/2) - (y_g(iy,dy,H0)-0.5*ly+dy/2)*(y_g(iy,dy,H0)-0.5*ly+dy/2) - (z_g(iz,dz,H0)-0.5*lz+dz/2)*(z_g(iz,dz,H0)-0.5*lz+dz/2)) for ix=1:size(H0,1), iy=1:size(H0,2), iz=1:size(H0,3)])
    Hold       = @ones(nx,ny,nz) .* H0
    H          = @ones(nx,ny,nz) .* H0
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
            @parallel compute_flux!(qHx, qHy, qHz, qHx2, qHy2, qHz2, H, D, θr_dτ, dx, dy, dz)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_update!(H, Hold, qHx, qHy, qHz, dτ_ρ, dt, dx, dy, dz)
                update_halo!(H)
            end
            iter += 1
            if iter % nout == 0
                @parallel check_res!(ResH, H, Hold, qHx2, qHy2, qHz2, dt, dx, dy, dz)
                err = norm_g(ResH) / sqrt(len_ResH_g)
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
            heatmap(Xi_g, Yi_g, H_v[:,:,z_sl]', dpi=150, aspect_ratio=1, framestyle=:box, xlims=(Xi_g[1],Xi_g[end]), ylims=(Yi_g[1],Yi_g[end]), xlabel="lx", ylabel="ly", c=:viridis, clims=(0,1), title="linear diffusion (nt=$it, iters=$ittot)")
            savefig("../../figures/diff_3D_lin_$(nx_g()).png")
        end
    end
    if me==0 && do_save
        !ispath("../../output") && mkdir("../../output")
        open("../../output/out_diff_3D_lin.txt","a") do io
            println(io, "$(nx_g()) $(ny_g()) $(nz_g()) $(ittot) $(nt)")
        end
    end
    if me==0 && do_save_viz
        !ispath("../../out_visu") && mkdir("../../out_visu")
        matwrite("../../out_visu/diff_3D_lin.mat", Dict("H_3D"=> Array(H_v), "xc_3D"=> Array(xc), "yc_3D"=> Array(yc), "zc_3D"=> Array(zc)); compress = true)
    end
    finalize_global_grid()
    return xc, yc, zc, H
end

if use_return
    xc, yc, zc, H = diffusion_3D_();
else
    diffusion_3D = begin diffusion_3D_(); return; end
end

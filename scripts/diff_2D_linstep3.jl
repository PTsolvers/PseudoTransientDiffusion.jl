const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const do_viz  = haskey(ENV, "DO_VIZ")  ? parse(Bool, ENV["DO_VIZ"])  : true
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : false
const do_save_viz = haskey(ENV, "DO_SAVE_VIZ") ? parse(Bool, ENV["DO_SAVE_VIZ"]) : false
const nx = haskey(ENV, "NX") ? parse(Int, ENV["NX"]) : 512
const ny = haskey(ENV, "NY") ? parse(Int, ENV["NY"]) : 512
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, LinearAlgebra, MAT

@parallel function compute_iter_params!(dt_ρ, D, Re, Vpdt, max_lxy)
    @all(dt_ρ) = Vpdt * max_lxy / @maxloc(D) / Re
    return
end

@parallel function compute_flux!(qHx, qHy, qHx2, qHy2, H, D, θr_dt, dx, dy)
    @all(qHx)  = (@all(qHx) * θr_dt - @av_xi(D) * @d_xi(H) / dx) / (1.0 + θr_dt)
    @all(qHy)  = (@all(qHy) * θr_dt - @av_yi(D) * @d_yi(H) / dy) / (1.0 + θr_dt)
    @all(qHx2) = -@av_xi(D) * @d_xi(H) / dx
    @all(qHy2) = -@av_yi(D) * @d_yi(H) / dy
    return
end

@parallel function compute_update!(H, Hold, qHx, qHy, dtauH, dt, dx, dy)
    @inn(H) = (@inn(H) + @all(dtauH) * (@inn(Hold) / dt - (@d_xa(qHx) / dx + @d_ya(qHy) / dy))) / (1.0 + @all(dtauH) / dt)
    return
end

@parallel function compute_update!(H, Hold, qHx, qHy, dt_ρ, dt, dx, dy)
    @inn(H) = (@inn(H) +  @all(dt_ρ) * (@inn(Hold) / dt - (@d_xa(qHx) / dx + @d_ya(qHy) / dy))) / (1.0 + @all(dt_ρ) / dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx2, qHy2, dt, dx, dy)
    @all(ResH) = -(@inn(H) - @inn(Hold)) / dt - (@d_xa(qHx2) / dx + @d_ya(qHy2) / dy)
    return
end

@parallel_indices (iy) function bc_x!(A)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function bc_y!(A)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

@views function diffusion_2D()
    # Physics
    lx, ly  = 10.0, 10.0    # domain size
    D1      = 1.0           # diffusion coefficient
    D2      = 1e-4          # diffusion coefficient
    ttot    = 1.0           # total simulation time
    dt      = 0.2           # physical time step
    # Numerics
    # nx, ny  = 2*256, 2*256  # numerical grid resolution
    tol     = 1e-8          # tolerance
    itMax   = 1e5           # max number of iterations
    nout    = 10            # tol check
    CFL     = 1/sqrt(2)     # CFL number
    # Derived numerics
    dx, dy  = lx / nx, ly / ny  # grid size    
    Vpdt    = CFL * min(dx, dy)
    max_lxy = max(lx, ly)
    Re      = π + sqrt(π^2 + (max_lxy^2 / max(D1,D2)) / dt)
    θr_dt   = max_lxy / Vpdt / Re
    xc, yc  = LinRange(-lx / 2, lx / 2, nx), LinRange(-ly / 2, ly / 2, ny)
    # Array allocation
    qHx     = @zeros(nx-1, ny-2)
    qHy     = @zeros(nx-2, ny-1)
    qHx2    = @zeros(nx-1, ny-2)
    qHy2    = @zeros(nx-2, ny-1)
    ResH    = @zeros(nx-2, ny-2)
    dt_ρ    = @zeros(nx-2, ny-2)
    # Initial condition
    D       = D1 * @ones(nx,ny)
    D[1:Int(ceil(nx / 2.2)),:] .= D2
    D[:,1:Int(ceil(ny / 2.2))] .= D2
    H0      = Data.Array(exp.(-xc.^2 .- (yc').^2))
    Hold    = @ones(nx,ny) .* H0
    H       = @ones(nx,ny) .* H0
    @parallel compute_iter_params!(dt_ρ, D, Re, Vpdt, max_lxy)
    t = 0.0; it = 0; ittot = 0; nt = Int(ceil(ttot / dt))
    # Physical time loop
    while it < nt
        iter = 0; err = 2 * tol
        # Pseudo-transient iteration
        while err > tol && iter < itMax
            @parallel compute_flux!(qHx, qHy, qHx2, qHy2, H, D, θr_dt, dx, dy)
            @parallel compute_update!(H, Hold, qHx, qHy, dt_ρ, dt, dx, dy)
            iter += 1
            if iter % nout == 0
                @parallel check_res!(ResH, H, Hold, qHx2, qHy2, dt, dx, dy)
                err = norm(ResH) / length(ResH)
            end
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz display(heatmap(xc, yc, Array(H'), aspect_ratio=1, framestyle=:box, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), xlabel="lx", ylabel="ly", c=:viridis, clims=(0, 1), title="linear step diffusion (nt=$it, iters=$ittot)")) end
    if do_save
        !ispath("../output") && mkdir("../output")
        open("../output/out_diff_2D_linstep3.txt","a") do io
            println(io, "$(nx) $(ny) $(ittot) $(nt)")
        end
    end
    if do_save_viz
        !ispath("../out_visu") && mkdir("../out_visu")
        matwrite("../out_visu/diff_2D_linstep3.mat", Dict("H_2D"=> Array(H), "xc_2D"=> Array(xc), "yc_2D"=> Array(yc)); compress = true)
    end
    return
end

diffusion_2D()

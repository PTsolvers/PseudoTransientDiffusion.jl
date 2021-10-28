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

@parallel function compute_flux!(qHx, qHy, qHx2, qHy2, H, D, θr_dτ, dx, dy)
    @all(qHx)  = (@all(qHx) * θr_dτ - D * @d_xi(H) / dx) / (1.0 + θr_dτ)
    @all(qHy)  = (@all(qHy) * θr_dτ - D * @d_yi(H) / dy) / (1.0 + θr_dτ)
    @all(qHx2) = -D * @d_xi(H) / dx
    @all(qHy2) = -D * @d_yi(H) / dy
    return
end

@parallel function compute_update!(H, Hold, qHx, qHy, dτ_ρ, dt, dx, dy)
    @inn(H) = (@inn(H) +  dτ_ρ * (@inn(Hold) / dt - (@d_xa(qHx) / dx + @d_ya(qHy) / dy))) / (1.0 + dτ_ρ / dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx2, qHy2, dt, dx, dy)
    @all(ResH) = -(@inn(H) - @inn(Hold)) / dt - (@d_xa(qHx2) / dx + @d_ya(qHy2) / dy)
    return
end

@views function diffusion_2D()
    # Physics
    lx, ly  = 10.0, 10.0    # domain size
    D       = 1.0           # diffusion coefficient
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
    Vpdτ    = CFL * min(dx, dy)
    Re      = π + sqrt(π^2 + (max(lx, ly)^2 / D / dt)) # Numerical Reynolds number
    θr_dτ   = max(lx, ly) / Vpdτ / Re
    dτ_ρ    = Vpdτ * max(lx, ly) / D / Re
    xc, yc  = LinRange(-lx / 2, lx / 2, nx), LinRange(-ly / 2, ly / 2, ny)
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
    t = 0.0; it = 0; ittot = 0; nt = Int(ceil(ttot/dt))
    # Physical time loop
    while it < nt
        iter = 0; err = 2 * tol
        # Pseudo-transient iteration
        while err > tol && iter < itMax
            @parallel compute_flux!(qHx, qHy, qHx2, qHy2, H, D, θr_dτ, dx, dy)
            @parallel compute_update!(H, Hold, qHx, qHy, dτ_ρ, dt, dx, dy)
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
    if do_viz display(heatmap(xc, yc, Array(H'), aspect_ratio=1, framestyle=:box, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), xlabel="lx", ylabel="ly", c=:viridis, clims=(0, 1), title="linear diffusion (nt=$it, iters=$ittot)")) end
    if do_save
        !ispath("../output") && mkdir("../output")
        open("../output/out_diff_2D_lin.txt","a") do io
            println(io, "$(nx) $(ny) $(ittot) $(nt)")
        end
    end
    if do_save_viz
        !ispath("../out_visu") && mkdir("../out_visu")
        matwrite("../out_visu/diff_2D_lin.mat", Dict("H_2D"=> Array(H), "xc_2D"=> Array(xc), "yc_2D"=> Array(yc)); compress = true)
    end
    return
end

diffusion_2D()

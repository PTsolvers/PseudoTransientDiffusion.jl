const use_return  = haskey(ENV, "USE_RETURN" ) ? parse(Bool, ENV["USE_RETURN"] ) : false
const USE_GPU     = haskey(ENV, "USE_GPU"    ) ? parse(Bool, ENV["USE_GPU"]    ) : false
const do_viz      = haskey(ENV, "DO_VIZ"     ) ? parse(Bool, ENV["DO_VIZ"]     ) : true
const do_save     = haskey(ENV, "DO_SAVE"    ) ? parse(Bool, ENV["DO_SAVE"]    ) : false
const do_save_viz = haskey(ENV, "DO_SAVE_VIZ") ? parse(Bool, ENV["DO_SAVE_VIZ"]) : false
const nx          = haskey(ENV, "NX"         ) ? parse(Int , ENV["NX"]         ) : 512
###
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end
using Plots, Printf, LinearAlgebra, MAT

@parallel function compute_flux!(qHx, qHx2, H, D, θr_dτ, dx)
    @all(qHx)  = (@all(qHx) * θr_dτ - D * @d(H) / dx) / (1.0 + θr_dτ)
    @all(qHx2) = -D * @d(H) / dx
    return
end

@parallel function compute_update!(H, Hold, qHx, dτ_ρ, dt, dx)
    @inn(H) = (@inn(H) +  dτ_ρ * (@inn(Hold) / dt - @d(qHx) / dx)) / (1.0 + dτ_ρ / dt)
    return
end

@parallel function check_res!(ResH, H, Hold, qHx2, dt, dx)
    @all(ResH) = -(@inn(H) - @inn(Hold)) / dt - @d(qHx2) / dx
    return
end

@views function diffusion_1D_()
    # Physics
    lx     = 10.0       # domain size
    D      = 1          # diffusion coefficient
    ttot   = 1.0        # total simulation time
    dt     = 0.2        # physical time step
    # Numerics
    # nx     = 2*256      # numerical grid resolution
    tol    = 1e-8       # tolerance
    itMax  = 1e5        # max number of iterations
    nout   = 10         # tol check
    CFL    = 0.99       # CFL number
    # Derived numerics
    dx     = lx / nx      # grid size
    Vpdτ   = CFL * dx
    Re     = π + sqrt(π^2 + (lx^2 / D / dt)) # Numerical Reynolds number
    θr_dτ  = lx / Vpdτ / Re
    dτ_ρ   = Vpdτ * lx / D / Re
    xc     = LinRange(dx/2, lx - dx/2, nx)
    # Array allocation
    qHx    = @zeros(nx-1)
    qHx2   = @zeros(nx-1)
    ResH   = @zeros(nx-2)
    dH     = @zeros(nx-2)
    # Initial condition
    H0     = Data.Array(exp.(-(xc .- lx/2).^2 / D))
    Hold   = @ones(nx) .* H0
    H      = @ones(nx) .* H0
    t = 0.0; it = 0; ittot = 0; nt = Int(ceil(ttot/dt))
    # Physical time loop
    while it < nt
        iter = 0; err = 2 * tol
        # Pseudo-transient iteration
        while err > tol && iter < itMax
            @parallel compute_flux!(qHx, qHx2, H, D, θr_dτ, dx)
            @parallel compute_update!(H, Hold, qHx, dτ_ρ, dt, dx)
            iter += 1
            if iter % nout == 0
                @parallel check_res!(ResH, H, Hold, qHx2, dt, dx)
                err = norm(ResH) / sqrt(length(ResH))
            end
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end
    # Analytic solution
    Hana = 1 / sqrt(4 * (ttot + 1 / 4)) * exp.(-(xc .- lx/2).^2 / (4 * D * (ttot + 1 / 4)))
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d, error vs analytic = %1.2e \n", round(ttot, sigdigits=2), it, nx, ittot, norm(Array(H) - Hana) / sqrt(nx))
    # Visualise
    if do_viz plot(xc, Array(H0), linewidth=3); display(plot!(xc, [Array(H) Array(Hana)], legend=false, framestyle=:box, linewidth=3, xlabel="lx", ylabel="H", title="linear diffusion (nt=$it, iters=$ittot)")) end
    if do_save
        !ispath("../output") && mkdir("../output")
        open("../output/out_diff_1D_lin.txt","a") do io
            println(io, "$(nx) $(ittot) $(nt)")
        end
    end
    if do_save_viz
        !ispath("../out_visu") && mkdir("../out_visu")
        matwrite("../out_visu/diff_1D_lin.mat", Dict("H_1D"=> Array(H), "xc_1D"=> Array(xc)); compress = true)
    end
    return xc, H
end

if use_return
    xc, H = diffusion_1D_();
else
    diffusion_1D = begin diffusion_1D_(); return; end
end

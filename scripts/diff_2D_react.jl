const use_return  = haskey(ENV, "USE_RETURN" ) ? parse(Bool, ENV["USE_RETURN"] ) : false
const USE_GPU     = haskey(ENV, "USE_GPU"    ) ? parse(Bool, ENV["USE_GPU"]    ) : false
const do_viz      = haskey(ENV, "DO_VIZ"     ) ? parse(Bool, ENV["DO_VIZ"]     ) : true
const nx          = haskey(ENV, "NX"         ) ? parse(Int , ENV["NX"]         ) : 512
const ny          = haskey(ENV, "NY"         ) ? parse(Int , ENV["NY"]         ) : 512
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, LinearAlgebra

@parallel function compute_flux!(qHx, qHy, qHx2, qHy2, H, D, θr_θkin, dx, dy)
    @all(qHx)  = (@all(qHx) * θr_θkin - D * @d_xi(H) / dx) / (1.0 + θr_θkin)
    @all(qHy)  = (@all(qHy) * θr_θkin - D * @d_yi(H) / dy) / (1.0 + θr_θkin)
    @all(qHx2) = -D * @d_xi(H) / dx
    @all(qHy2) = -D * @d_yi(H) / dy
    return
end

@parallel function compute_update!(H, Heq, qHx, qHy, θkin_ρ, θkin, dx, dy)
    @inn(H) = (@inn(H) +  θkin_ρ * (@inn(Heq) / θkin - (@d_xa(qHx) / dx + @d_ya(qHy) / dy))) / (1.0 + θkin_ρ / θkin)
    return
end

@parallel function check_res!(ResH, H, Heq, qHx2, qHy2, θkin, dx, dy)
    @all(ResH) = -(@inn(H) - @inn(Heq)) / θkin - (@d_xa(qHx2) / dx + @d_ya(qHy2) / dy)
    return
end

@views function diffusion_react_2D_()
    # Physics
    lx, ly  = 10.0, 10.0    # domain size
    D       = 1.0           # diffusion coefficient
    θkin    = 0.1           # characteristic time of reaction
    # Numerics
    # nx, ny  = 2*256, 2*256  # numerical grid resolution
    tol     = 1e-8          # tolerance
    itMax   = 1e5           # max number of iterations
    nout    = 10            # tol check
    CFL     = 1/sqrt(2)     # CFL number
    Da      = π + sqrt(π^2 + (lx^2 / D / θkin)) # Numerical Reynolds number
    # Derived numerics
    dx, dy  = lx / nx, ly / ny  # grid size    
    Vpdτ    = CFL * min(dx, dy)
    θr_θkin = max(lx, ly) / Vpdτ / Da
    θkin_ρ  = Vpdτ * max(lx, ly) / D / Da
    xc, yc  = LinRange(dx/2, lx - dx/2, nx), LinRange(dy/2, ly - dy/2, ny)
    # Array allocation
    qHx     = @zeros(nx - 1,ny - 2)
    qHy     = @zeros(nx - 2,ny - 1)
    qHx2    = @zeros(nx - 1,ny - 2)
    qHy2    = @zeros(nx - 2,ny - 1)
    ResH    = @zeros(nx - 2,ny - 2)
    # Initial condition
    H0      = Data.Array(exp.(-(xc .- lx/2).^2 .- ((yc .- ly/2)').^2))
    Heq     = @ones(nx,ny) .* H0
    H       = @zeros(nx,ny)
    # Time loop
    iter = 0; err = 2 * tol
    # Pseudo-transient iteration
    while err > tol && iter < itMax
        @parallel compute_flux!(qHx, qHy, qHx2, qHy2, H, D, θr_θkin, dx, dy)
        @parallel compute_update!(H, Heq, qHx, qHy, θkin_ρ, θkin, dx, dy)
        iter += 1
        if iter % nout == 0
            @parallel check_res!(ResH, H, Heq, qHx2, qHy2, θkin, dx, dy)
            err = norm(ResH) / sqrt(length(ResH))
        end
    end
    if isnan(err) error("NaN") end
    @printf("nx = %d, ny = %d, iterations tot = %d \n", nx, ny, iter)
    # Visualise
    if do_viz display(heatmap(xc, yc, Array(H'), aspect_ratio=1, framestyle=:box, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), xlabel="lx", ylabel="ly", c=:viridis, clims=(0, 1), title="linear diffusion-reaction (iters=$iter)")) end
    return xc, yc, H
end

if use_return
    xc, yc, H = diffusion_react_2D_();
else
    diffusion_react_2D = begin diffusion_react_2D_(); return; end
end

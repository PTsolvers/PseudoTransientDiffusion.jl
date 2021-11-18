using Plots, Printf, LinearAlgebra

@views av(A) = 0.5 .* (A[1:end-1] .+ A[2:end])

@views function diffusion_1D(; implicit=false, do_viz=true)
    # Physics
    lx     = 10.0       # domain size
    ttot   = 1.0        # total simulation time
    dt     = 0.2        # physical time step
    # Numerics
    nx     = 512        # numerical grid resolution
    tol    = 1e-8       # tolerance
    itMax  = 1e4        # max number of iterations
    nout   = 10         # tol check
    CFL    = 0.8        # CFL number: 1.0 implicit, 0.8 explicit 
    ε      = 1e-2       # small number to avoid division by 0
    # Derived numerics
    dx     = lx / nx    # grid size
    Vpdτ   = CFL * dx
    xc     = LinRange(dx / 2, lx - dx / 2, nx)
    # Array allocation
    qHx    = zeros(nx-1)
    qHx2   = zeros(nx-1)
    ResH   = zeros(nx-2)
    Re     = zeros(nx  )
    θr_dτ  = zeros(nx-1)
    dτ_ρ   = zeros(nx-2)
    # Initial condition
    H0     = exp.(-(xc .- lx / 2).^2)
    Hold   = ones(nx) .* H0
    H      = ones(nx) .* H0
    D      = H.^3
    t = 0.0; it = 0; ittot = 0; nt = Int(ceil(ttot / dt))
    # Physical time loop
    while it < nt
        iter = 0; err = 2 * tol
        # Pseudo-transient iteration
        while err > tol && iter < itMax
            # Diffusion coefficient
            D          .= H.^3
            # Assign iter params
            Re         .= π .+ sqrt.(π^2 .+ lx^2 ./ max.(D,ε) ./ dt)
            θr_dτ      .= lx ./ Vpdτ ./ av(Re)
            dτ_ρ       .= Vpdτ .* lx ./ max.(D[2:end-1],ε) ./ Re[2:end-1]
            # PT updates
            if implicit
                qHx        .= (qHx .* θr_dτ .- av(D) .* diff(H) ./ dx) ./ (1.0 .+ θr_dτ)
                H[2:end-1] .= (H[2:end-1] .+  dτ_ρ .* (Hold[2:end-1] ./ dt .- diff(qHx) ./ dx)) ./ (1.0 .+ dτ_ρ ./ dt)
            else
                qHx        .= qHx        .+  1.0 ./ θr_dτ .* ( .-qHx .-av(D) .* diff(H) ./ dx )
                H[2:end-1] .= H[2:end-1] .+  dτ_ρ  .* ( .-(H[2:end-1] .- Hold[2:end-1]) ./ dt .- diff(qHx) ./ dx)
            end
            iter += 1
            # Check error (explicit residual)
            if iter % nout == 0
                qHx2 .= .-av(D) .* diff(H) ./ dx
                ResH .= .-(H[2:end-1] .- Hold[2:end-1]) ./ dt .- diff(qHx2) ./ dx
                err = norm(ResH) / sqrt(length(ResH))
            end
        end
        # Update H
        Hold .= H
        ittot += iter; it += 1; t += dt
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
    # Visualise
    if do_viz plot(xc, Array(H0), linewidth=3); display(plot!(xc, Array(H), legend=false, framestyle=:box, linewidth=3, xlabel="lx", ylabel="H", title="nonlinear diffusion (nt=$it, iters=$ittot)")) end
    return
end

diffusion_1D()

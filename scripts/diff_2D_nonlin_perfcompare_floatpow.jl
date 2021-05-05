const USE_GPU = true
const USE_PARALLEL = true
const USE_PARALLEL_INDICES = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
#using Plots, Printf, Statistics, LinearAlgebra  # ATTENTION: plotting fails inside plotting library if using flag '--math-mode=fast'.

@static if USE_PARALLEL_INDICES
    macro qHx(ix,iy) esc(:((g_mu/3.0)*_dx*((H[$ix,$iy] + H[$ix+1,$iy  ])*0.5)^3.5 * ((H[$ix+1,$iy  ]-H[$ix, $iy]) + (B[$ix+1,$iy  ]-B[$ix,$iy])))) end  # @all(qHx) = g_mu/3.0*_dx*@av_xa(H)^3*(@d_xa(H)+@d_xa(B))
    macro qHy(ix,iy) esc(:((g_mu/3.0)*_dy*((H[$ix,$iy] + H[$ix,  $iy+1])*0.5)^3.5 * ((H[$ix,  $iy+1]-H[$ix, $iy]) + (B[$ix,  $iy+1]-B[$ix,$iy])))) end  # @all(qHy) = g_mu/3.0*_dy*@av_ya(H)^3*(@d_ya(H)+@d_ya(B))
    macro dτ() esc(:((1.0/(1.0/dt+1.0/((((min(dx^2,dy^2)/g_mu)*3.0)/H[ix+1,iy+1]^3)/4.1))))) end
end




#        macro dτ() esc(:(1.0/(1.0/dt + 1.0/(min(dx^2,dy^2)/g_mu*3.0/(@inn(H)^3)/4.1)))) end



@static if USE_PARALLEL
    @static if USE_PARALLEL_INDICES
        @parallel_indices (ix,iy) function update_H!(H2::Data.Array, H::Data.Array, dHdτ::Data.Array, H_t::Data.Array, B::Data.Array, g_mu::Data.Number, _dx::Data.Number, _dy::Data.Number, dx::Data.Number, dy::Data.Number, dt::Data.Number, damp::Data.Number)
            if ix <= size(dHdτ,1) && iy <= size(dHdτ,2)
                dHdτ[ix,iy] = -((H[ix+1,iy+1] - H_t[ix+1,iy+1]))*(1.0/dt) + (   # @all(dHdτ) = -(@inn(H)-@inn(H_t))*(1.0/dt) + (
                                  _dx*(@qHx(ix+1,iy+1) - @qHx(ix  ,iy+1))       #                  _dx*@d_xi(qHx)
                                + _dy*(@qHy(ix+1,iy+1) - @qHy(ix+1,iy  ))       #                + _dy*@d_yi(qHy)
                              )                                                 #               )
                              + damp*dHdτ[ix,iy]                                #               + damp*@all(dHdτ)
            end
            if ix <= size(H2,1)-2 && iy <= size(H2,2)-2
                H2[ix+1,iy+1] = H[ix+1,iy+1] + @dτ() * dHdτ[ix,iy]               # @inn(H)    = @inn(H) + @dτ()*@all(dHdτ)
            end
            return
        end
    else
        @parallel function compute_flux!(qHx::Data.Array, qHy::Data.Array, H::Data.Array, B::Data.Array, g_mu::Data.Number, _dx::Data.Number, _dy::Data.Number)
            @all(qHx) = g_mu/3.0*_dx*@av_xa(H)^3.5*(@d_xa(H)+@d_xa(B))
            @all(qHy) = g_mu/3.0*_dy*@av_ya(H)^3.5*(@d_ya(H)+@d_ya(B))
            return
        end


        @parallel function update_H!(dHdτ::Data.Array, H::Data.Array, H_t::Data.Array, qHx::Data.Array, qHy::Data.Array, g_mu::Data.Number, _dx::Data.Number, _dy::Data.Number, dx::Data.Number, dy::Data.Number, dt::Data.Number, damp::Data.Number)
            @all(dHdτ) = -(@inn(H)-@inn(H_t))*(1.0/dt) + (_dx*@d_xi(qHx) + _dy*@d_yi(qHy)) + damp*@all(dHdτ)
            @inn(H)    = @inn(H) + @dτ()*@all(dHdτ)
            return
        end
    end
else
    @views  d_xa(A) = A[2:end  , :     ] .- A[1:end-1, :     ];
    @views  d_xi(A) = A[2:end  ,2:end-1] .- A[1:end-1,2:end-1];
    @views  d_ya(A) = A[ :     ,2:end  ] .- A[ :     ,1:end-1];
    @views  d_yi(A) = A[2:end-1,2:end  ] .- A[2:end-1,1:end-1];
    @views  d_zi(A) = A[2:end-1,2:end-1] .- A[2:end-1,2:end-1];
    @views av_xa(A) = (A[1:end-1, :] .+ A[2:end, :]).*0.5
    @views av_ya(A) = (A[:,1:end-1] .+ A[:,2:end]).*0.5
    @views   inn(A) = A[2:end-1,2:end-1]

    @views function compute_flux!(qHx::Data.Array, qHy::Data.Array, H::Data.Array, B::Data.Array, g_mu::Data.Number, _dx::Data.Number, _dy::Data.Number)
        qHx .= g_mu./3.0.*_dx.*av_xa(H).^3 .*(d_xa(H).+d_xa(B))
        qHy .= g_mu./3.0.*_dy.*av_ya(H).^3 .*(d_ya(H).+d_ya(B))
        return
    end

    @views function update_H!(dHdτ::Data.Array, H::Data.Array, H_t::Data.Array, qHx::Data.Array, qHy::Data.Array, g_mu::Data.Number, _dx::Data.Number, _dy::Data.Number, dx::Data.Number, dy::Data.Number, dt::Data.Number, damp::Data.Number)
        dHdτ               .= .-(inn(H).-inn(H_t)).*(1.0./dt) .+ (_dx.*d_xi(qHx) .+ _dy.*d_yi(qHy)) .+ damp.*dHdτ
        H[2:end-1,2:end-1] .= inn(H) .+ (1.0./(1.0./dt .+ 1.0./(min(dx.^2,dy.^2)./g_mu.*3.0./(inn(H).^3)./4.1))) .* dHdτ
        return
    end
end

##################################################
@views function sia2D()
# Physics
g_mu     = 1.0
lx, ly   = 2.0, 2.0

# Numerics
nx, ny   = 8192, 8192
nt       = 10 #2 #10
itermax  = 100
ncheck   = 10
tolnl    = 1e-8
damp     = 0.8
dx       = lx/(nx-1)
dy       = ly/(ny-1)
_dx, _dy = 1.0/dx, 1.0/dy

# Array initializations
B       = @zeros(nx  ,ny  )
H       = @zeros(nx  ,ny  )
H_t     = @zeros(nx  ,ny  )
#H_τ     = @zeros(nx  ,ny  )
dHdτ    = @zeros(nx-2,ny-2)
@static if USE_PARALLEL_INDICES
    H2  = @zeros(nx  ,ny  )
else
    qHx = @zeros(nx-1,ny  )
    qHy = @zeros(nx  ,ny-1)
end

# Initial conditions
B      .= 0.1 #Data.Array([0.1 + 0.1*exp(-(((ix-1)*dx-lx/2)*3)^2-(((iy-1)*dy-ly/2)*3)^2) for ix=1:size(B,1), iy=1:size(B,2)])
H      .= 1.1 #Data.Array([0.1 + 1.0*exp(-(((ix-1)*dx-lx/2)*4)^2-(((iy-1)*dy-ly/2)*4)^2) for ix=1:size(H,1), iy=1:size(H,2)])

# Preparation of visualisation
# gr() #pyplot()
# ENV["GKSwstype"]="nul"
# anim = Animation();

# Time loop
dt = 400*min(dx^2,dy^2)/g_mu*3.0/maximum(H)^3/4.1 # factor*CFL
niter=0; t=0.0
for it = 1:nt
    if (it==2) global time_tic = Base.time(); niter = 0 end
    #@printf("it = %d, max H = %1.3e \n", it, maximum(H))
    #heatmap(0:dx:lx, 0:dy:ly, transpose(Array(H)); aspect_ratio=1, xlims=(0, lx), ylims=(0, ly), c=:viridis, title="H"); frame(anim)  #Simpler: heatmap!(transpose(Array(H)); aspect_ratio=1, c=:viridis, title="H"); frame(anim)
    #H_t .= H
    err=2.0*tolnl; iter=1
    while err > tolnl && iter <= itermax
        # if (iter % ncheck == 0) H_τ .= H; end
        @static if USE_PARALLEL
            @static if USE_PARALLEL_INDICES
                @parallel update_H!(H2, H, dHdτ, H_t, B, g_mu, _dx, _dy, dx, dy, dt, damp)
                H, H2 = H2, H;
            else
                @parallel compute_flux!(qHx, qHy, H, B, g_mu, _dx, _dy)
                @parallel update_H!(dHdτ, H, H_t, qHx, qHy, g_mu, _dx, _dy, dx, dy, dt, damp)
            end
        else
            compute_flux!(qHx, qHy, H, B, g_mu, _dx, _dy)
            update_H!(dHdτ, H, H_t, qHx, qHy, g_mu, _dx, _dy, dx, dy, dt, damp)
        end
        # if iter % ncheck == 0
        #     err = mean(abs.((H.-H_τ)))
        #     @printf("   iter = %d, err = %1.3e \n", iter, err)
        # end
        iter+=1; niter+=1
    end
    t = t + dt
end
time_s = Base.time() - time_tic;

# Performance
A_eff = (2*2+2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B and H_t have to be read (H_t for pseudo-transient method): 2 whole-array memaccess)
##A_eff = (4*2+2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 2 whole-array memaccess)
t_it  = time_s/niter                           # Execution time per iteration [s]
T_eff = A_eff/t_it                             # Effective memory throughput [GB/s]
println("time_s=$time_s T_eff=$T_eff t=$t");

# Postprocessing
#gif(anim, "sia2D.gif", fps = 15)
#mp4(anim, "sia2D.mp4", fps = 15)

end

sia2D()

using Plots

include("diff_1D_lin.jl")
# include("diff_1D_lin2.jl")
# include("diff_1D_linstep.jl")
# include("diff_1D_linstep2.jl")
# include("diff_1D_nonlin.jl")
# include("diff_1D_nonlin2.jl")

@views function runtests()

    resol = 16 * 2 .^ (1:5)

    out = zeros(2, length(resol))

    for i = 1:length(resol)
        res = resol[i]
        nxx, iter = diffusion_1D(; nx=res);
        out[1,i] = nxx
        out[2,i] = iter
    end

    fontsize = 10
    opts = (yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
            linewidth=3, markersize=5, markershape=:circle, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", 
            xlabel="resolution nx", ylabel="ittot/nx" )

    p1=plot(out[1,:], out[2,:]./out[1,:]; opts...)
      # plot!(resol, out[2,:]./resol; opts...)
    display(plot(p1))
    
end


@time runtests()

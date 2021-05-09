using Plots, JLD

function vizualise(data)
    fontsize = 10
    opts = (yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"), label=name, 
            linewidth=3, markersize=5, markershape=:circle, framestyle=:box,
            titlefontsize=fontsize, titlefont="Courier", 
            legendfontsize=fontsize, legendfont="Courrier", foreground_color_legend=nothing, 
            xlabel="resolution nx", ylabel="ittot/nx" )

    p1=plot(data[1,:], data[end,:]./data[1,:]; opts...)
    # plot!(resol, data[2,:]./resol; opts...)
    display(plot(p1))
    return
end

@views function runtests_1D(; do_viz=true, do_save=true)
    
    resol = 16 * 2 .^ (1:10)

    out = zeros(2, length(resol))

    for i = 1:length(resol)
        res = resol[i]
        nxx, iter = diffusion_1D(; nx=res)
        out[1,i] = nxx
        out[2,i] = iter
    end

    if do_viz vizualise(out) end

    if do_save
        !ispath("../output") && mkdir("../output")
        save("../output/out_$(name).jld", "out", out)
    end
end

@views function runtests_2D(; do_viz=true, do_save=true)
    
    resol = 16 * 2 .^ (1:8)

    out = zeros(3, length(resol))

    for i = 1:length(resol)
        res = resol[i]
        nxx, nyy, iter = diffusion_2D(; nx=res, ny=res)
        out[1,i] = nxx
        out[2,i] = nyy
        out[3,i] = iter
    end

    if do_viz vizualise(out) end

    if do_save
        !ispath("../output") && mkdir("../output")
        save("../output/out_$(name).jld", "out", out)
    end
end
# - Run 1D scaling ----------------------------------------------
# include("diff_1D_lin.jl"); name = "diff_1D_lin"
# include("diff_1D_lin2.jl"); name = "diff_1D_lin2"
# include("diff_1D_linstep.jl"); name = "diff_1D_linstep"
# include("diff_1D_linstep2.jl"); name = "diff_1D_linstep2"
# include("diff_1D_nonlin.jl"); name = "diff_1D_nonlin"
# include("diff_1D_nonlin2.jl"); name = "diff_1D_nonlin2"

# @time runtests_1D(; do_save=false)

# - Run 2D scaling ----------------------------------------------
include("diff_2D_lin.jl"); name = "diff_2D_lin"
# include("diff_2D_lin2.jl"); name = "diff_2D_lin2"

@time runtests_2D(; do_save=true)

# - Run 3D scaling ----------------------------------------------



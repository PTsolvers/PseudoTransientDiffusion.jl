using Plots, JLD

function vizualise(data, name, iviz)
    fontsize = 10
    opts = (yaxis=(font(fontsize, "Courier")), xaxis=(:log10, font(fontsize, "Courier")), label=name, 
            linewidth=3, markersize=5, markershape=:circle, framestyle=:box, 
            titlefontsize=fontsize, titlefont="Courier", 
            legendfontsize=fontsize, legendfont="Courrier", foreground_color_legend=nothing, 
            xlabel="resolution nx", ylabel="ittot/nx" )
    if iviz ==1
        p1=plot((data[1,:]), data[end,:]./data[1,:]; opts...)
    else
        p1=plot!((data[1,:]), data[end,:]./data[1,:]; opts...)
    end
    display(plot(p1, size=(500,300), dpi=150))
    return
end


# test_name = ("diff_1D_lin","diff_1D_lin2",)
# test_name = ("diff_1D_linstep","diff_1D_linstep2",)
# test_name = ("diff_1D_nonlin", "diff_1D_nonlin2",)

# test_name = ("diff_2D_lin", "diff_2D_lin2",)
# test_name = ("diff_2D_linstep", "diff_2D_linstep2",)
# test_name = ("diff_2D_nonlin", "diff_2D_nonlin2",)

# test_name = ("diff_1D_lin", "diff_2D_lin",)
# test_name = ("diff_1D_lin2", "diff_2D_lin2",)
# test_name = ("diff_1D_linstep", "diff_2D_linstep",)
# test_name = ("diff_1D_linstep2", "diff_2D_linstep2",)
# test_name = ("diff_1D_nonlin", "diff_2D_nonlin",)
# test_name = ("diff_1D_nonlin2", "diff_2D_nonlin2",)

test_name = ("diff_1D_nonlin", "diff_2D_nonlin","diff_1D_nonlin2", "diff_2D_nonlin2",)

for iviz = 1:length(test_name)
    
    name = test_name[iviz]
    
    out = load("../output/out_$(name).jld", "out")

    vizualise(out, name, iviz)
end

# savefig("../figures/fig_$(test_name[1]).png")

# savefig("../figures/fig_1D_2D_$(test_name[1]).png")

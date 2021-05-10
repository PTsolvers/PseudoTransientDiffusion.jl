using Plots, JLD

function vizualise(data, name)
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

@views function runtests_1D(name; gen_data=false, do_save=false, do_viz=true)

    if gen_data
        resol = 16 * 2 .^ (1:10)

        out = zeros(2, length(resol))

        for i = 1:length(resol)
            res = resol[i]
            nxx, iter = diffusion_1D(; nx=res)
            out[1,i] = nxx
            out[2,i] = iter
        end

        if do_save
            !ispath("../output") && mkdir("../output")
            save("../output/out_$(name).jld", "out", out)
        end
    else
        out = load("../output/out_$(name).jld", "out")
    end

    if do_viz vizualise(out, name) end
end

@views function runtests_2D(name; gen_data=false, do_save=false, do_viz=true)
    
    if gen_data
        resol = 16 * 2 .^ (1:6)

        out = zeros(3, length(resol))

        for i = 1:length(resol)
            res = resol[i]
            nxx, nyy, iter = diffusion_2D(; nx=res, ny=res)
            out[1,i] = nxx
            out[2,i] = nyy
            out[3,i] = iter
        end

        if do_save
            !ispath("../output") && mkdir("../output")
            save("../output/out_$(name).jld", "out", out)
        end
    else
        out = load("../output/out_$(name).jld", "out")
    end

    if do_viz vizualise(out, name) end
end

# - Run tests ----------------------------------------------
tests = "test_1D"

if tests=="test_1D"
    
    test_name = ("diff_1D_lin", "diff_1D_lin2", "diff_1D_linstep", "diff_1D_linstep2", "diff_1D_nonlin", "diff_1D_nonlin2")
    
    for itest = 1:length(test_name)
        testfile = test_name[itest] * ".jl"; print(">--- "); println(testfile)
        include(testfile)
        @time runtests_1D(test_name[itest]; gen_data=true, do_save=true)
    end

elseif tests=="test_2D"

    test_name = ("diff_2D_lin", "diff_2D_lin2", "diff_2D_linstep", "diff_2D_linstep2", "diff_2D_nonlin", "diff_2D_nonlin2")

    for itest = 1:length(test_name)
        testfile = test_name[itest] * ".jl"; print(">--- "); println(testfile)
        include(testfile)
        @time runtests_2D(test_name[itest];  gen_data=true, do_save=true)
    end

elseif tests=="test_3D"

end


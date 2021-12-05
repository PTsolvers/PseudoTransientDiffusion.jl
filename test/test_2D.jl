using Test, ReferenceTests, BSON
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil

ENV["USE_GPU"] = false
ENV["DO_VIZ"]  = false
ENV["DO_SAVE"] = false
ENV["DO_SAVE_VIZ"] = false
ENV["NX"] = 64
ENV["NY"] = 64
ENV["USE_RETURN"] = true

# Reference test using ReferenceTests.jl
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2) for (v1,v2) in zip(values(d1), values(d2))])

## 2D tests
include("../scripts/diff_2D_lin.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d1  = Dict(:X=> xc[indsx], :H=>H[indsx,indsy])

include("../scripts/diff_2D_linstep.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d2  = Dict(:X=> xc[indsx], :H=>H[indsx,indsy])

include("../scripts/diff_2D_nonlin.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d3  = Dict(:X=> xc[indsx], :H=>H[indsx,indsy])

include("../scripts/diff_2D_nonlin_perf.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d4  = Dict(:X=> xc[indsx], :H=>H[indsx,indsy])

include("../scripts/diff_2D_react.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d5  = Dict(:X=> xc[indsx], :H=>H[indsx,indsy])

@testset "Reference-tests diffusion 2D" begin
    @test_reference "reftest-files/test_diff_2D_lin.bson" d2d1 by=comp
    @test_reference "reftest-files/test_diff_2D_linstep.bson" d2d2 by=comp
    @test_reference "reftest-files/test_diff_2D_nonlin.bson" d2d3 by=comp
    @test_reference "reftest-files/test_diff_2D_nonlin_perf.bson" d2d4 by=comp
    @test_reference "reftest-files/test_diff_2D_react.bson" d2d5 by=comp
end

@reset_parallel_stencil()

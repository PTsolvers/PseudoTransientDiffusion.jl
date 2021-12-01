using Test, ReferenceTests, BSON
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil

ENV["USE_GPU"] = false
ENV["DO_VIZ"]  = false
ENV["DO_SAVE"] = false
ENV["DO_SAVE_VIZ"] = false
ENV["NX"] = 64
ENV["USE_RETURN"] = true

# # Unit tests
# @testset "Unit-tests" begin
#     # some unit tests
# end

# Reference test using ReferenceTests.jl
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2) for (v1,v2) in zip(values(d1), values(d2))])

## 1D tests
@reset_parallel_stencil()
include("../scripts/diff_1D_lin.jl")
inds = Int.(ceil.(LinRange(1, length(xc), 12)))
d1d1 = Dict(:X=> xc[inds], :H=>H[inds])

@reset_parallel_stencil()
include("../scripts/diff_1D_linstep.jl")
inds = Int.(ceil.(LinRange(1, length(xc), 12)))
d1d2 = Dict(:X=> xc[inds], :H=>H[inds])

@reset_parallel_stencil()
include("../scripts/diff_1D_nonlin.jl")
inds = Int.(ceil.(LinRange(1, length(xc), 12)))
d1d3 = Dict(:X=> xc[inds], :H=>H[inds])

@reset_parallel_stencil()
include("../scripts/diff_1D_react.jl")
inds = Int.(ceil.(LinRange(1, length(xc), 12)))
d1d4 = Dict(:X=> xc[inds], :H=>H[inds])

@testset "Reference-tests diffusion 1D" begin
    @test_reference "reftest-files/test_diff_1D_lin.bson" d1d1 by=comp
    @test_reference "reftest-files/test_diff_1D_linstep.bson" d1d2 by=comp
    @test_reference "reftest-files/test_diff_1D_nonlin.bson" d1d3 by=comp
    @test_reference "reftest-files/test_diff_1D_react.bson" d1d4 by=comp
end

@reset_parallel_stencil()

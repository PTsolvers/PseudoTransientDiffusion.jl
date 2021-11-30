using PseudoTransientDiffusion
using Test, ReferenceTests, BSON

ENV["USE_GPU"] = false
ENV["DO_VIZ"]  = true
ENV["DO_SAVE"] = false
ENV["DO_SAVE_VIZ"] = false
ENV["NX"] = 32
ENV["NY"] = 32
ENV["NZ"] = 32

# include("../scripts/diff_3D_lin.jl")

# # Unit tests
# @testset "Unit-tests" begin
#     @test size(inn(ones(3,3,3))) == (1,1,1)
#     @test inn(ones(3,3,3))[1] == 1
#     H = ones(3,3,3)
#     @parallel compute_update!(H, ones(3,3,3), 1.0)
#     @test H[2,2,2] == 2
#     @test sum(H.==1)==26
# end

# # Reference test using ReferenceTests.jl
# comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-5) for (v1,v2) in zip(values(d1), values(d2))])
# inds = Int.(ceil.(LinRange(1, length(Xc_g), 12)))
# d = Dict(:X=> Xc_g[inds], :H_g=>H_g[inds])

# @testset "Ref-file" begin
#     @test_reference "reftest-files/test.bson" d by=comp
# end

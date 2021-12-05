
include("./shared.jl")

## 3D tests
include("../scripts/diff_3D/diff_3D_nonlin_multixpu_perf.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
indsz = Int.(ceil.(LinRange(1, length(zc), 12)))
d3d   = Dict(:X=> xc[indsx], :H=>H[indsx,indsy,indsz])

@testset "Reference-tests diffusion 3D nonlin perf" begin
    @test_reference "reftest-files/test_diff_3D_nonlin_multixpu_perf.bson" d3d by=comp
end

@reset_parallel_stencil()

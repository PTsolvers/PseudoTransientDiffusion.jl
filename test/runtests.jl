# NOTE: This file contains many parts that are copied from the file runtests.jl from the Package ParallelStencil.jl.
push!(LOAD_PATH, "../src")

using PseudoTransientDiffusion

function runtests()
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir = pwd()
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = sort(filter(istest, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...)))

    nfail = 0
    printstyled("Testing PseudoTransientDiffusion.jl\n"; bold=true, color=:white)

    for f in testfiles
        try
            run(`$exename -O3 --startup-file=no --check-bounds=no $(joinpath(testdir, f))`)
        catch ex
            nfail += 1
        end
    end
    return nfail
end

exit(runtests())

import Pkg
Pkg.activate(".")
Pkg.add(["Plots", "HDF5", "StaticArrays", "OffsetArrays", "FastChebInterp"])
Pkg.add(url="https://github.com/JuliaFolds/ParallelMagics.jl")
Pkg.develop(path="..")
Pkg.instantiate()
Pkg.precompile()

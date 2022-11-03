import Pkg
Pkg.activate(".")
Pkg.develop(path="..")
Pkg.add(url="https://github.com/JuliaFolds/ParallelMagics.jl")
Pkg.instantiate()
Pkg.precompile()

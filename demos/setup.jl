import Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/JuliaFolds/ParallelMagics.jl")
Pkg.develop(path="..")
Pkg.instantiate()
Pkg.precompile()

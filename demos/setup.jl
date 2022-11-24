import Pkg
Pkg.activate(".")
Pkg.add(["Plots", "HDF5", "StaticArrays", "OffsetArrays", "FastChebInterp"])
Pkg.develop(path="..")
Pkg.instantiate()
Pkg.precompile()

import Pkg
Pkg.activate(".")
Pkg.add(["Plots", "ProgressBars", "StaticArrays", "OffsetArrays"])
Pkg.develop(path="..")
Pkg.instantiate()
Pkg.precompile()

import Pkg
Pkg.activate(".")
Pkg.dev("..")
Pkg.instantiate()
Pkg.precompile()
import Pkg
Pkg.activate(".")
Pkg.develop(path="..")
Pkg.instantiate()
Pkg.precompile()

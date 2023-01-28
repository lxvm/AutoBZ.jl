"""
    AutoBZ

A package implementing iterated-adaptive integration and equispace integration
specialized for Brillouin-zone integration of localized and broadened
integrands, respectively, using the algorithms described by [Kaye et
al.](http://arxiv.org/abs/2211.12959). For tree-adaptive integration, see
[HCubature.jl](https://github.com/JuliaMath/HCubature.jl). The package also
provides utilities for user-defined integrands based on Fourier series.
"""
module AutoBZ

using LinearAlgebra

using StaticArrays
using Polyhedra
using QuadGK
# using FFTW

include("IntegrationLimits.jl")
include("iterated_integration.jl")
include("bz.jl")
include("equispace_integration.jl")
include("FourierSeries.jl")
include("integrands.jl")
include("integrators.jl")

include("Jobs.jl")

end
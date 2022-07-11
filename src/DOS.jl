module DOS

using LinearAlgebra

using StaticArrays
using OffsetArrays
using HCubature
using QuadGK
using FFTW

include("linalg.jl")
include("FourierSeries.jl")
include("limits.jl")
include("integrands.jl")
include("adaptive_integration.jl")
include("equispace_integration.jl")
include("irreducible_BZ.jl")
include("sweeps.jl")

end
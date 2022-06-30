module DOS

using LinearAlgebra

using StaticArrays
using OffsetArrays
using HCubature
using QuadGK
using FFTW

include("inv.jl")
include("FourierSeries.jl")
include("adaptive_integration.jl")
include("equispace_integration.jl")
include("irreducible_BZ.jl")

end
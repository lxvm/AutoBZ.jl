module DOS

using LinearAlgebra

using StaticArrays
using OffsetArrays
using HCubature
using QuadGK

include("FourierSeries.jl")
include("adaptive_integration.jl")
include("irreducible_BZ.jl")

end
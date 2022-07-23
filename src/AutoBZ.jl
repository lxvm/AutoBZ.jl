"""
    AutoBZ

A small package implementing adaptive iterated integration and equispace
integration for Brillouin-zone integration of sharply-peaked functions.
"""
module AutoBZ

using LinearAlgebra

using StaticArrays
using HCubature
using QuadGK

include("IntegrationLimits.jl")
include("callback.jl")
include("adaptive_integration.jl")
include("equispace_integration.jl")

"""
    Applications

A small module depending on AutoBZ that calculates DOS and optical conductivity.
Note: it could be a separate package, but is included as an example application.
"""
module Applications

using LinearAlgebra

using StaticArrays
using OffsetArrays
using FFTW

using  ..AutoBZ: IntegrationLimits, CubicLimits
import ..AutoBZ: lower, upper, rescale

# include("linalg.jl")
include("FourierSeries.jl")
include("limits.jl")
include("irreducible_BZ.jl")
include("integrands.jl")
include("sweeps.jl")

end

end
"""
    AutoBZCore

A small module implementing iterated-adaptive integration and equispace
integration specialized for Brillouin-zone integration of localized and
broadened integrands, respectively, using the algorithms described by [Kaye et
al.](http://arxiv.org/abs/2211.12959). For tree-adaptive integration, see
[HCubature.jl](https://github.com/JuliaMath/HCubature.jl).
"""
module AutoBZCore

using LinearAlgebra
using StaticArrays
using QuadGK

include("IntegrationLimits.jl")
include("iterated_integration.jl")
include("equispace_integration.jl")

end
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
include("adaptive_integration.jl")
include("equispace_integration.jl")

include("AdaptChebInterp.jl")
include("EquiBaryInterp.jl")

"""
    Applications

A small module depending on AutoBZ that calculates density of states and optical
conductivity, and also provides tools for custom integrands evaluated by Wannier
interpolation.
"""
module Applications

using LinearAlgebra

using StaticArrays
using OffsetArrays
using Combinatorics: permutations
# using FFTW

using ..EquiBaryInterp: LocalEquiBaryInterp

using  ..AutoBZ: IntegrationLimits, CubicLimits, 
    equispace_integration, automatic_equispace_integration, discretize_equispace_,
    iterated_integration, alloc_segbufs
import ..AutoBZ: box, lower, upper, nsyms, symmetries,
    equispace_pre_eval, equispace_npt_update, evaluate_integrand,
    iterated_pre_eval, infer_f

include("linalg.jl")
include("FourierSeries.jl")
include("band_velocities.jl")
include("fourier3d.jl")
include("self_energies.jl")
include("limits.jl")
include("fermi.jl")
include("integrands.jl")
include("custom_equispace.jl")
include("custom_adaptive.jl")
include("evaluators.jl")
include("wannier90io.jl")
include("self_energies_io.jl")


end

include("Jobs.jl")

end
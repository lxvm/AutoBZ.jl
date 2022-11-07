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

"""
    evaluate_integrand(f, x)

By default, this calls `f(x)`, however the caller may dispatch on the type of
`f` if they would like to specialize this function together with
`equispace_pre_eval` so that `x` is a more useful precomputation (e.g. a Fourier
series evaluated at a grid point).

This function is also called once at the beginning of an `iterated_integration`
call in order to infer the type of the range of `f` assuming `f` is type-stable.
"""
evaluate_integrand(f, x) = f(x)

include("IntegrationLimits.jl")
include("adaptive_integration.jl")
include("equispace_integration.jl")

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

using  ..AutoBZ: IntegrationLimits, CubicLimits, 
    equispace_integration, automatic_equispace_integration, discretize_equispace_
import ..AutoBZ: box, lower, upper, nsyms, symmetries,
    equispace_pre_eval, equispace_npt_update, evaluate_integrand,
    iterated_pre_eval, infer_f

# include("linalg.jl")
include("FourierSeries.jl")
include("band_velocities.jl")
include("self_energies.jl")
include("limits.jl")
include("integrands.jl")
include("custom_equispace.jl")
include("custom_adaptive.jl")
include("wannier90io.jl")

end

end
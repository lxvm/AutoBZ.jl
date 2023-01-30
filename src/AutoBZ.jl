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
using Polyhedra: Hull, HyperPlane, intersect, points, fulldim
using QuadGK: quadgk, do_quadgk, alloc_segbuf

import Polyhedra: coefficient_type


# Below is a comprehensive list of exports and the files in which they are defined


# component 1: Generic iterated adaptive integration (IAI)

export AbstractLimits, limits, domain_type
export CubicLimits, PolyhedralLimits, CompositeLimits
include("AbstractLimits.jl")

export AbstractIteratedIntegrand, nvars, iterated_pre_eval, iterated_integrand
export ThunkIntegrand, IteratedIntegrand, AssociativeOpIntegrand
include("AbstractIteratedIntegrand.jl")

export iterated_integration # the main routine
export iterated_tol_update, iterated_segs, iterated_inference, iterated_integral_type
include("iterated_integration.jl")


# component 2: Brillouin zones and symmetrized equispace integration / periodic trapezoidal rule (PTR)

export AbstractBZ, FullBZ, IrreducibleBZ
export basis, nsyms, symmetries, symmetrize, limits, boundingbox, vol, domain_type
include("AbstractBZ.jl")

export equispace_integration, automatic_equispace_integration # main routines
export equispace_npt_update, equispace_rule, equispace_rule!, equispace_integrand, equispace_evalrule
include("equispace_integration.jl")


# component 3: Fourier series with IAI & PTR optimizations

export AbstractFourierSeries, period, contract, value, coefficients, coefficient_type, fourier_type, phase_type
export FourierSeries, FourierSeriesDerivative, OffsetFourierSeries, ManyFourierSeries, ManyOffsetsFourierSeries
export fourier_kernel, fourier_kernel!, fourier_rule!
include("AbstractFourierSeries.jl")

export AbstractFourierIntegrand, finner, ftotal, series, params
export FourierIntegrand, IteratedFourierIntegrand
include("AbstractFourierIntegrand.jl")


# component 4: convenient interface to integration routines

export AbstractIntegrator, quad_integrand, quad_routine, quad_args, quad_kwargs, limits
export IteratedIntegrator, FourierIntegrator
include("AbstractIntegrator.jl")


# component 5: Submodule containing applications

include("Jobs.jl")


end
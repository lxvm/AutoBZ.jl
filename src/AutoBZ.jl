"""
An applications package implementing iterated-adaptive integration and equispace
integration for electronic structure and transport calculations. It excels at
integrating both localized and also broadened Brillouin-zone integrands, using
the algorithms described by [Kaye et al.](http://arxiv.org/abs/2211.12959). This
package also provides multi-threaded routines for parallelized calculations.
See `AutoBZCore` if you only need the essential functionality of the library to
define custom BZ integrands.
"""
module AutoBZ

using LinearAlgebra
using Printf

using HDF5
using StaticArrays
using QuadGK: quadgk


using AutoSymPTR
using IteratedIntegration
using FourierSeriesEvaluators
using AutoBZCore

import FourierSeriesEvaluators: period, contract!, evaluate

export read_h5_to_nt, write_nt_to_h5
export adaptive_fourier_integration, automatic_equispace_fourier_integration, equispace_fourier_integration, auto_fourier_integration
export run_dos_adaptive, run_dos_auto_equispace, run_dos_equispace, run_dos
export run_kinetic_adaptive, run_kinetic_auto_equispace, run_kinetic_equispace, run_kinetic



"""
    AbstractWannierInterp{gauge,N} <: AbstractInplaceFourierSeries{N}

An abstract subtype of `AbstractInplaceFourierSeries` representing in-place
Fourier series evaluators for Wannier-interpolated quantities with a choice of
basis, or `gauge`.
"""
abstract type AbstractWannierInterp{gauge,N,T} <: AbstractInplaceFourierSeries{N,T} end

gauge(::AbstractWannierInterp{G}) where G = G


include("linalg.jl")
include("hamiltonian.jl")
include("band_velocities.jl")
include("self_energies.jl")
include("self_energies_io.jl")
include("wannier90io.jl")
include("fermi.jl")
include("apps.jl")
include("jobs.jl")

end
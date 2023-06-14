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
using LinearAlgebra: checksquare

using StaticArrays
using Reexport
@reexport using FourierSeriesEvaluators
@reexport using AutoBZCore

import FourierSeriesEvaluators: period, deriv, offset, shift,
    contract, evaluate, coefficients, show_details
import AutoBZCore: SymRep, symmetrize_, AutoBZAlgorithm


using EquiBaryInterp: LocalEquiBaryInterp
using BaryRational: aaa
using HChebInterp: hchebinterp

export AbstractBZ, FBZ, IBZ, HBZ, CubicSymIBZ
export AbstractSelfEnergy, lb, ub
export AbstractWannierInterp
export AbstractGauge, Wannier, Hamiltonian
export AbstractGaugeInterp, gauge
export AbstractCoordinate, Lattice, Cartesian
export AbstractCoordInterp, coord
export AbstractVelocityComponent, Whole, Inter, Intra
export AbstractVelocityInterp, vcomp, hamiltonian, shift!
include("definitions.jl")

export diag_inv, tr_inv, tr_mul, herm, commutator
include("linalg.jl")

export fermi, fermi′, fermi_window, fermi_window_limits
include("fermi.jl")

export HamiltonianInterp, BerryConnectionInterp, GradientVelocityInterp, CovariantVelocityInterp
include("interp.jl")

export EtaSelfEnergy, ConstScalarSelfEnergy, ScalarSelfEnergy, MatrixSelfEnergy
include("self_energies.jl")

export GlocIntegrand, DiagGlocIntegrand, TrGlocIntegrand, DOSIntegrand
export TransportFunctionIntegrand, TransportDistributionIntegrand
export KineticCoefficientIntegrand, OpticalConductivityIntegrand
export ElectronDensityIntegrand
export AuxTransportDistributionIntegrand, AuxKineticCoefficientIntegrand, AuxOpticalConductivityIntegrand
include("apps.jl")

export load_bz
include("bzkinds.jl")

export load_self_energy
include("self_energies_io.jl")

export load_interp, load_wannier90_data
include("wannier90io.jl")


end

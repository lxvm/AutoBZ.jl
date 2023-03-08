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

!isdefined(Base, :get_extension) && using Requires
@static if !isdefined(Base, :get_extension)
    function __init__()
        @require SymmetryReduceBZ = "49a35663-c880-4242-bebb-1ec8c0fa8046" include("../ext/SymmetryReduceBZExt.jl")
    end
end

using LinearAlgebra
using LinearAlgebra: checksquare
using Printf: @sprintf

using StaticArrays
using HDF5
using Reexport
@reexport using AutoBZCore

import AutoBZCore: SymRep, batchsolve, FourierIntegrand,
    FourierSeriesEvaluators.period, FourierSeriesEvaluators.contract!, FourierSeriesEvaluators.evaluate, FourierSeriesEvaluators.coefficients, FourierSeriesEvaluators.show_details,
    AutoSymPTR.npt_update


export AbstractSelfEnergy, lb, ub
export AbstractWannierInterp, gauge, hamiltonian, shift!
export AbstractVelocity, vcomp
include("definitions.jl")

export diag_inv, tr_inv, tr_mul, herm, commutator
include("linalg.jl")

export fermi, fermi′, fermi_window, fermi_window_limits
include("fermi.jl")

export Hamiltonian
include("hamiltonian.jl")

export HamiltonianVelocity, CovariantHamiltonianVelocity
include("velocities.jl")

export load_bz
include("bzkinds.jl")

export EtaSelfEnergy, ConstScalarSelfEnergy, ScalarSelfEnergy, MatrixSelfEnergy
include("self_energies.jl")

export load_self_energy
include("self_energies_io.jl")

export load_wannier90_data
include("wannier90io.jl")

export GlocIntegrand, DiagGlocIntegrand, TrGlocIntegrand, DOSIntegrand
export TransportFunctionIntegrand, TransportDistributionIntegrand
export KineticCoefficientIntegrand, OpticalConductivityIntegrand
export ElectronDensityIntegrand
include("apps.jl")

include("jobs.jl")

end
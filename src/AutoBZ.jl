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
        @require Brillouin = "23470ee3-d0df-4052-8b1a-8cbd6363e7f0" begin
            @require PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a" include("../ext/BrillouinPlotlyJSExt.jl")
        end
    end
end

using LinearAlgebra
using LinearAlgebra: checksquare

using StaticArrays
using Reexport
@reexport using AutoBZCore

import AutoBZCore: SymRep, symmetrize_,
    Integrand, FourierIntegrand, construct_problem,
    FourierSeriesEvaluators.period, FourierSeriesEvaluators.contract,
    FourierSeriesEvaluators.evaluate, FourierSeriesEvaluators.coefficients,
    FourierSeriesEvaluators.show_details, FourierSeriesEvaluators.deriv,
    FourierSeriesEvaluators.offset, FourierSeriesEvaluators.shift,
    AutoSymPTR.npt_update


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

export HamiltonianInterp
include("hamiltonian.jl")

export BerryConnectionInterp
include("berry.jl")

export GradientVelocityInterp, CovariantVelocityInterp
include("velocities.jl")

export EtaSelfEnergy, ConstScalarSelfEnergy, ScalarSelfEnergy, MatrixSelfEnergy
include("self_energies.jl")

export GlocIntegrand, DiagGlocIntegrand, TrGlocIntegrand, DOSIntegrand
export TransportFunctionIntegrand, TransportDistributionIntegrand
export KineticCoefficientIntegrand, OpticalConductivityIntegrand
export ElectronDensityIntegrand
include("apps.jl")

export load_bz
include("bzkinds.jl")

export load_self_energy
include("self_energies_io.jl")

export load_interp, load_wannier90_data
include("wannier90io.jl")


end
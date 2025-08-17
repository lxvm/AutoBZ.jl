module AutoBZAuxQuadGKExt

using AuxQuadGK
using AutoBZ: AbstractCoordSymRep, SymmetricBZ, AbstractSelfEnergy, IntegralAlgorithm, AbstractVelocityInterp, AutoBZAlgorithm, LinearSystemAlgorithm, JLInv, lb, ub, _DynamicalTransportDistributionSolver, _trG_auxfun
import AutoBZ: symmetrize_, AuxKineticCoefficientSolver

function symmetrize_(rep::AbstractCoordSymRep, bz::SymmetricBZ, x::AuxValue)
    val = symmetrize_(rep, bz, x.val)
    aux = symmetrize_(rep, bz, x.aux)
    return AuxValue(val, aux)
end

function AuxKineticCoefficientSolver(auxfun::F, Σ::AbstractSelfEnergy, fdom, falg::IntegralAlgorithm, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, linalg::LinearSystemAlgorithm=JLInv(); kws...) where {F}
    _DynamicalTransportDistributionSolver((Γ, h, v, sol) -> AuxValue(Γ, auxfun(v, sol.G1, sol.G2)), Σ, fdom, falg, hv, bz, bzalg, linalg; kws...)
end
function AuxKineticCoefficientSolver(Σ::AbstractSelfEnergy, fdom::Tuple, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    AuxKineticCoefficientSolver(_trG_auxfun, Σ, fdom, falg, hv, bz, bzalg, linalg; kws...)
end
function AuxKineticCoefficientSolver(auxfun::F, Σ::AbstractSelfEnergy, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, linalg::LinearSystemAlgorithm=JLInv(); kws...) where {F}
    AuxKineticCoefficientSolver(auxfun, Σ, (lb(Σ), ub(Σ)), falg, hv, bz, bzalg, linalg; kws...)
end
function AuxKineticCoefficientSolver(Σ::AbstractSelfEnergy, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    AuxKineticCoefficientSolver(_trG_auxfun, Σ, falg, hv, bz, bzalg, linalg; kws...)
end

function AuxKineticCoefficientSolver(auxfun::F, h::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, Σ::AbstractSelfEnergy, fdom, falg, linalg::LinearSystemAlgorithm=JLInv(); kws...) where {F}
    _DynamicalTransportDistributionSolver((Γ, h, v, sol) -> AuxValue(Γ, auxfun(v, sol.G1, sol.G2)), h, bz, bzalg, Σ, fdom, falg, linalg; kws...)
end
function AuxKineticCoefficientSolver(h::AbstractVelocityInterp, bz, bzalg, Σ::AbstractSelfEnergy, fdom, falg, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    AuxKineticCoefficientSolver(_trG_auxfun, h, bz, bzalg, Σ, fdom, falg, linalg; kws...)
end
function AuxKineticCoefficientSolver(auxfun::F, h::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, Σ::AbstractSelfEnergy, falg, linalg::LinearSystemAlgorithm=JLInv(); kws...) where {F}
    AuxKineticCoefficientSolver(auxfun, h, bz, bzalg, Σ, (lb(Σ), ub(Σ)), falg, linalg; kws...)
end
function AuxKineticCoefficientSolver(h::AbstractVelocityInterp, bz, bzalg, Σ::AbstractSelfEnergy, falg, linalg::LinearSystemAlgorithm=JLInv(); kws...)
    AuxKineticCoefficientSolver(_trG_auxfun, h, bz, bzalg, Σ, falg, linalg; kws...)
end
end
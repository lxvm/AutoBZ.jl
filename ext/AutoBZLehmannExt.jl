module AutoBZLehmannExt

using AutoBZ
using AutoBZ: AbstractHamiltonianInterp, TraceInverseAlgorithm
using Lehmann

import AutoBZ: ElectronDensitySolver

function ElectronDensitySolver(Σ::AbstractSelfEnergy, falg::DLRIntegral, h::AbstractHamiltonianInterp, bz, bzalg, trinvalg::TraceInverseAlgorithm=JLTrInv(); kws...)
    error("TODO")
end
function ElectronDensitySolver(h::AbstractHamiltonianInterp, bz, bzalg, Σ::AbstractSelfEnergy, falg::DLRIntegral, trinvalg::TraceInverseAlgorithm=JLTrInv(); kws...)
    error("TODO")
end
end

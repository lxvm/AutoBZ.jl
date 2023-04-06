module BrillouinPlotlyJSExt

using LinearAlgebra: checksquare

if isdefined(Base, :get_extension)
    using AutoBZ: Hamiltonian, HamiltonianInterp
    using Brillouin
    using PlotlyJS
    import PlotlyJS: plot
else
    using ..AutoBZ: Hamiltonian, HamiltonianInterp
    using ..Brillouin
    using ..PlotlyJS
    import ..PlotlyJS: plot
end

function plot(kpi::KPathInterpolant, h::HamiltonianInterp{Hamiltonian()}, args...; kwargs...)
    data = h.(kpi)
    nband = checksquare(eltype(h))
    bands = [ map(h_k -> h_k.values[i], data) for i in 1:nband ]
    plot(kpi, bands, args...; kwargs...)
end

end

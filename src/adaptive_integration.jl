export tree_integration, iterated_integration

tree_integration(f, a, b; kwargs...) = hcubature(f, a, b; kwargs...)

"""
Contract the outermost index of the Fourier Series
"""
contract(f::DOSIntegrand, x) = DOSIntegrand(contract(f.ϵ, x), f.ω, f.η, f.μ)
function contract(f::FourierSeries{N}, x) where {N}
    C = f.coeffs
    ϕ = 2π*im*first(x)/last(f.period)
    C′ = mapreduce(+, CartesianIndices(C); dims=N) do i
        C[i]*exp(last(i.I)*ϕ)
    end
    FourierSeries(reshape(C′, axes(C)[1:N-1]), pop(f.period))
end

iterated_integration(f, a, b; kwargs...) = iterated_integration(f, SVector(a), SVector(b); kwargs...)
iterated_integration(f, a::SVector{1}, b::SVector{1}; kwargs...) = hcubature(f, a, b; kwargs...)
function iterated_integration(f, a::SVector, b::SVector; kwargs...)
    hcubature(SVector(last(a)), SVector(last(b)); kwargs...) do x
        g = contract(f, x)
        first(iterated_integration(g, pop(a), pop(b); kwargs...))
    end
end
#=
iterated_integration(f, a::SVector{1}, b::SVector{1}; kwargs...) = hquadrature(f, first(a), first(b); kwargs...)
function iterated_integration(f, a::SVector, b::SVector; kwargs...)
    hquadrature(last(a), last(b); kwargs...) do x
        g = contract(f, x)
        first(iterated_integration(g, pop(a), pop(b); kwargs...))
    end
end

iterated_integration(f, a::SVector{1}, b::SVector{1}; kwargs...) = quadgk(f, first(a), first(b); kwargs...)
function iterated_integration(f, a::SVector, b::SVector; kwargs...)
    quadgk(last(a), last(b); kwargs...) do x
        g = contract(f, x)
        first(iterated_integration(g, pop(a), pop(b); kwargs...))
    end
end
=#
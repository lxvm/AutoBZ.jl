export tree_integration, iterated_integration

tree_integration(f, a, b; kwargs...) = hcubature(f, a, b; kwargs...)

"""
Contract the outermost index of the Fourier Series
"""
contract(f, x::SVector{1}) = contract(f, first(x))
contract(f::DOSIntegrand, x) = DOSIntegrand(contract(f.ϵ, x), f.ω, f.η, f.μ)
function contract(f::FourierSeries{N}, x::Number)  where {N}
    C = f.coeffs
    ϕ = 2π*im*x/last(f.period)
    C′ = mapreduce(+, CartesianIndices(C); dims=N) do i
        C[i]*exp(last(i.I)*ϕ)
    end
    FourierSeries(reshape(C′, axes(C)[1:N-1]), pop(f.period))
end
# performance hack for larger tensors that allocates less
function contract(f::FourierSeries{3}, x::Number)
    N=3
    C = contract(f.coeffs, first(x) / last(f.period))
    j = SVector{N-1}(1:N-1)
    FourierSeries(C, f.period[j])
end
function contract(C::AbstractArray{<:Any,3}, ϕ::Number)
    N = 3 # the function body works for any N>1
    -2first(axes(C,N))+1 == size(C,N) || throw("array indices are not of form -n:n")
    ax = CartesianIndices(axes(C)[1:N-1])
    @inbounds r = C[ax, 0]
    if size(C,N) > 1
        z₀ = exp(2π*im*ϕ)
        z = one(z₀)
        for i in 1:last(axes(C,N))
            z *= z₀
            @inbounds r += z*view(C, ax, i) + conj(z)*view(C, ax, -i)
        end
    end
    r
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
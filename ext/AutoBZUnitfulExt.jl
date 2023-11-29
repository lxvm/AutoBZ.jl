module AutoBZUnitfulExt

using Unitful
using LinearAlgebra
using StaticArrays
import AutoBZ: _inv,_eigen

# https://github.com/PainterQubits/Unitful.jl/issues/538

function _inv(x::StaticMatrix{N,M,T}) where {N,M,T <: Unitful.AbstractQuantity}
    m = _inv(map(ustrip, x))
    iq = eltype(m)
    map(Quantity{iq, inv(dimension(T)), typeof(inv(unit(T)))}, m)
end

function _eigen(x::Hermitian{T,<:StaticMatrix}) where {T<:Quantity}
    y = Hermitian(map(ustrip, x.data))
    e = _eigen(y)
    iq = eltype(e.values)
    return Eigen(map(Quantity{iq, dimension(T), typeof(unit(T))}, e.values), e.vectors)
end
end

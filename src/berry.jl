struct BerryConnectionInterp{B,G,N,T,A,TB} <: AbstractCoordInterp{B,G,N,T}
    a::A
    B::TB
    BerryConnectionInterp{B}(a::A, b::TB) where {B,A<:ManyFourierSeries,TB} =
        new{B,GaugeDefault(BerryConnectionInterp),ndims(a),eltype(a),A,TB}(a, b)
end

function BerryConnectionInterp(a, B; coord=CoordDefault(BerryConnectionInterp))
    BerryConnectionInterp{coord}(a, B)
end

period(bc::BerryConnectionInterp) = period(bc.a)

# we cannot change these properties from Wannier90 input
GaugeDefault(::Type{<:BerryConnectionInterp}) = Wannier()
CoordDefault(::Type{<:BerryConnectionInterp}) = Cartesian()

contract(bc::BerryConnectionInterp, x::Number, ::Val{d}) where d =
    BerryConnectionInterp{coord(bc)}(contract(bc.a, x, Val(d)), bc.B)

function evaluate(bc::BerryConnectionInterp, x::NTuple{1})
    as = map(herm, evaluate(bc.a, x)) # take Hermitian part of A since Wannier90 may not enforce it
    return to_coord(bc, bc.B, SVector(as))
end
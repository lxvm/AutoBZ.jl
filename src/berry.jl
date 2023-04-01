struct BerryConnectionInterp{P,B,G,N,T,A,TB} <: AbstractCoordInterp{B,G,N,T}
    a::A
    B::TB
    BerryConnectionInterp{P,B}(a::A, b::TB) where {P,B,A<:ManyFourierSeries,TB} =
        new{P,B,GaugeDefault(BerryConnectionInterp),ndims(a),eltype(a),A,TB}(a, b)
end

function BerryConnectionInterp{P}(a, B; coord=CoordDefault(BerryConnectionInterp{P})) where P
    BerryConnectionInterp{P,coord}(a, B)
end

period(bc::BerryConnectionInterp) = period(bc.a)

GaugeDefault(::Type{<:BerryConnectionInterp}) = Wannier()
CoordDefault(::Type{<:BerryConnectionInterp{P}}) where P = P

contract(bc::BerryConnectionInterp, x::Number, ::Val{d}) where d =
    BerryConnectionInterp{CoordDefault(bc),coord(bc)}(contract(bc.a, x, Val(d)), bc.B)

function evaluate(bc::BerryConnectionInterp, x::NTuple{1})
    as = map(herm, evaluate(bc.a, x)) # take Hermitian part of A since Wannier90 may not enforce it
    return to_coord(bc, bc.B, SVector(as))
end
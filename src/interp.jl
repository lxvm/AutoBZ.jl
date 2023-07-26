"""
    HamiltonianInterp(f::FourierSeries; gauge=:Wannier)

A wrapper for `FourierSeries` with an additional gauge that allows for
convenient diagonalization of the result. For details see [`to_gauge`](@ref).
"""
struct HamiltonianInterp{G,N,T,F} <: AbstractHamiltonianInterp{G,N,T}
    f::F
    HamiltonianInterp{G}(f::F) where {G,F<:FourierSeries} =
        new{G,ndims(f),eltype(f),F}(f)
end

HamiltonianInterp(f; gauge=GaugeDefault(HamiltonianInterp)) =
    HamiltonianInterp{gauge}(f)

contract(h::HamiltonianInterp, x::Number, ::Val{d}) where d =
    HamiltonianInterp{gauge(h)}(contract(h.f, x, Val(d)))

GaugeDefault(::Type{<:HamiltonianInterp}) = Wannier()

period(h::HamiltonianInterp) = period(h.f)

evaluate(h::HamiltonianInterp, x::NTuple{1}) =
    to_gauge(h, evaluate(h.f, x))

coefficients(h::HamiltonianInterp) = coefficients(h.f)

# ------------------------------------------------------------------------------

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
CoordDefault(::Type{<:BerryConnectionInterp}) = Cartesian()
CoordDefault(::Type{<:BerryConnectionInterp{P}}) where P = P

contract(bc::BerryConnectionInterp, x::Number, ::Val{d}) where d =
    BerryConnectionInterp{CoordDefault(bc),coord(bc)}(contract(bc.a, x, Val(d)), bc.B)

function evaluate(bc::BerryConnectionInterp, x::NTuple{1})
    as = map(herm, evaluate(bc.a, x)) # take Hermitian part of A since Wannier90 may not enforce it
    return to_coord(bc, bc.B, SVector(as))
end

coefficients(b::BerryConnectionInterp) = coefficients(b.a)
# ------------------------------------------------------------------------------

raise_multiplier(::Val{0}) = Val(1)
raise_multiplier(::Val{1}) = Val(2)
raise_multiplier(a) = a + 1

struct GradientVelocityInterp{C,B,G,N,T,H,V,TA} <: AbstractVelocityInterp{C,B,G,N,T}
    h::H
    v::V
    A::TA
    function GradientVelocityInterp{C,B,G}(h::H, v::V, A::TA) where {C,B,G,H<:AbstractHamiltonianInterp,V,TA}
        @assert gauge(h) == GaugeDefault(H)
        return new{C,B,G,ndims(h),eltype(h),H,V,TA}(h, v, A)
    end
end

function GradientVelocityInterp(h::AbstractHamiltonianInterp, A;
    gauge=GaugeDefault(GradientVelocityInterp),
    coord=CoordDefault(GradientVelocityInterp),
    vcomp=VcompDefault(GradientVelocityInterp))
    GradientVelocityInterp{vcomp,coord,gauge}(h, (), A)
end

hamiltonian(hv::GradientVelocityInterp) = hv.h
CoordDefault(::Type{<:GradientVelocityInterp}) = Lattice()
GaugeDefault(::Type{<:GradientVelocityInterp}) = Wannier()
VcompDefault(::Type{<:GradientVelocityInterp}) = Whole()

function contract(hv::GradientVelocityInterp{C,B,G}, x::Number, ::Val{d}) where {C,B,G,d}
    h = contract(hv.h, x, Val(d))
    tpx = 2pi*x
    v = map(vi -> contract(vi, tpx, Val(d)), hv.v)
    # compute the derivative of the current dimension
    dv = deriv(hv.h.f)
    vd = contract(FourierSeries(hv.h.f.c, period=map(x->2pi*x, period(hv.h.f)), deriv=ntuple(n -> n == d ? raise_multiplier(dv[d]) : dv[n], Val(ndims(hv))), offset=offset(hv.h.f), shift=shift(hv.h.f)), tpx, Val(d))
    GradientVelocityInterp{C,B,G}(h, (vd, v...), hv.A)
end

function evaluate(hv::GradientVelocityInterp, x::NTuple{1})
    tpx = (2pi*x[1],)   # period adjusted to correct for angular coordinates in derivative
    v = map(vi -> evaluate(vi, tpx), hv.v)
    # compute the derivative of the current dimension
    v1 = evaluate(FourierSeries(hv.h.f.c, period=(2pi*period(hv.h.f)[1]), deriv=(raise_multiplier(deriv(hv.h.f)[1]),), offset=offset(hv.h.f), shift=shift(hv.h.f)), tpx)
    h, vs = to_vcomp_gauge(hv, evaluate(hv.h, x),  (v1, v...))
    return (h, to_coord(hv, hv.A, SVector(vs)))
end

"""
    covariant_velocity(H, Hα, Aα)

Evaluates the velocity operator ``\\hat{v}_{\\alpha} = -\\frac{i}{\\hbar}
[\\hat{r}_{\\alpha}, \\hat{H}]`` with the following expression, equivalent to
eqn. 18 in [Yates et al.](https://doi.org/10.1103/PhysRevB.75.195121)
```math
\\hat{v}_{\\alpha} = \\frac{1}{\\hbar} \\hat{H}_{\\alpha} + \\frac{i}{\\hbar} [\\hat{H}, \\hat{A}_{\\alpha}]
```
where the ``\\alpha`` index implies differentiation by ``k_{\\alpha}``. Note
that the terms that correct the derivative of the band velocity
Also, this function takes ``\\hbar = 1``.
"""
covariant_velocity(H, ∂H_∂α, ∂A_∂α) = ∂H_∂α + (im*I)*commutator(H, ∂A_∂α)


"""
    CovariantVelocityInterp(hv::GradientVelocityInterp{Val(:Wannier)}, a::ManyFourierSeries)

Uses the Berry connection to return fully gauge-covariant velocities. Returns a
tuple of the Hamiltonian and the three velocity matrices.
"""
struct CovariantVelocityInterp{C,B,G,N,T,HV,A} <: AbstractVelocityInterp{C,B,G,N,T}
    hv::HV
    a::A
    function CovariantVelocityInterp{C,B,G}(hv::HV, a::A) where {C,B,G,HV<:GradientVelocityInterp,A<:BerryConnectionInterp{B}}
        @assert vcomp(hv) == VcompDefault(HV)
        @assert gauge(hv) == GaugeDefault(HV)
        @assert coord(hv) == B
        return new{C,B,G,ndims(hv),eltype(hv),HV,A}(hv, a)
    end
end
function CovariantVelocityInterp(hv, a;
    gauge=GaugeDefault(CovariantVelocityInterp),
    coord=CoordDefault(CovariantVelocityInterp),
    vcomp=VcompDefault(CovariantVelocityInterp))
    @assert period(hv) == period(a)
    @assert ndims(hv) == length(a.a.fs)
    # @assert coord(hv) == coord(a) # function and keyword name conflict
    # TODO convert hv and A to the requested coordinate instead of throwing
    # method error in inner constructor
    CovariantVelocityInterp{vcomp,coord,gauge}(hv, a)
end

hamiltonian(chv::CovariantVelocityInterp) = hamiltonian(chv.hv)

CoordDefault(::Type{<:CovariantVelocityInterp}) = Lattice()
GaugeDefault(::Type{<:CovariantVelocityInterp}) = Wannier()
VcompDefault(::Type{<:CovariantVelocityInterp}) = Whole()

contract(chv::CovariantVelocityInterp{C,B,G}, x::Number, ::Val{d}) where {C,B,G,d} =
    CovariantVelocityInterp{C,B,G}(contract(chv.hv, x, Val(d)), contract(chv.a, x, Val(d)))

function evaluate(chv::CovariantVelocityInterp, x::NTuple{1})
    hw, vws = evaluate(chv.hv, x)
    as = evaluate(chv.a, x)
    to_vcomp_gauge(chv, hw, map((v, a) -> covariant_velocity(hw, v, a), vws, as))
    # Note that we already enforced the final coordinate in the inner constructor
end

# some special methods for inferring the rule
# for HamiltonianInterp
function Base.zero(::Type{FourierValue{X,S}}) where {X,A,B,C,S<:Eigen{A,B,C}}
    return FourierValue(zero(X),eigen(Hermitian(zero(C))))
end
# for the velocity interps
function Base.zero(::Type{FourierValue{X,S}}) where {X,W,V,S<:Tuple{W,V}}
    return FourierValue(zero(X),(zero(W), zero(V)))
end
function Base.zero(::Type{FourierValue{X,S}}) where {X,A,B,C,D<:Eigen{A,B,C},V,S<:Tuple{D,V}}
    return FourierValue(zero(X),(eigen(Hermitian(zero(C))), zero(V)))
end

# ----------------


struct MassVelocityInterp{C,B,G,N,T,H,V,M,TA} <: AbstractVelocityInterp{C,B,G,N,T}
    h::H
    v::V
    m::M
    A::TA
    function MassVelocityInterp{C,B,G}(h::H, v::V, m::M, A::TA) where {C,B,G,H<:AbstractHamiltonianInterp,V,M,TA}
        @assert gauge(h) == GaugeDefault(H)
        return new{C,B,G,ndims(h),eltype(h),H,V,M,TA}(h, v, m, A)
    end
end

function MassVelocityInterp(h::AbstractHamiltonianInterp, A;
    gauge=GaugeDefault(GradientVelocityInterp),
    coord=CoordDefault(GradientVelocityInterp),
    vcomp=VcompDefault(GradientVelocityInterp))
    MassVelocityInterp{vcomp,coord,gauge}(h, (), (), A)
end

hamiltonian(hv::MassVelocityInterp) = hv.h
CoordDefault(::Type{<:MassVelocityInterp}) = Lattice()
GaugeDefault(::Type{<:MassVelocityInterp}) = Wannier()
VcompDefault(::Type{<:MassVelocityInterp}) = Whole()

function contract(hv::MassVelocityInterp{C,B,G}, x::Number, ::Val{d}) where {C,B,G,d}
    h = contract(hv.h, x, Val(d))
    tpx = 2pi*x
    v = map(vi -> contract(vi, tpx, Val(d)), hv.v)
    # compute the derivative of the current dimension
    dv = deriv(hv.h.f)
    vd = contract(FourierSeries(hv.h.f.c, period=map(x->2pi*x, period(hv.h.f)), deriv=ntuple(n -> n == d ? raise_multiplier(dv[d]) : dv[n], Val(ndims(hv))), offset=offset(hv.h.f), shift=shift(hv.h.f)), tpx, Val(d))
    # compute second derivative of the current dimension
    m = map(y -> map(x -> contract(x, tpx, Val(d)), y), hv.m)
    vdd = map(vi -> contract(FourierSeries(vi.c, period=period(vi), deriv=ntuple(n -> n == d ? raise_multiplier(dv[d]) : dv[n], Val(ndims(hv))), offset=offset(hv.h.f), shift=shift(hv.h.f)), tpx, Val(d)), hv.v)
    vd2 = contract(FourierSeries(hv.h.f.c, period=map(x->2pi*x, period(hv.h.f)), deriv=ntuple(n -> n == d ? raise_multiplier(raise_multiplier(dv[d])) : dv[n], Val(ndims(hv))), offset=offset(hv.h.f), shift=shift(hv.h.f)), tpx, Val(d))
    MassVelocityInterp{C,B,G}(h, (vd, v...), ((vd2, vdd...), m...), hv.A)
end

function evaluate(hv::MassVelocityInterp, x::NTuple{1})
    tpx = (2pi*x[1],)   # period adjusted to correct for angular coordinates in derivative
    v = map(vi -> evaluate(vi, tpx), hv.v)
    # compute the derivative of the current dimension
    v1 = evaluate(FourierSeries(hv.h.f.c, period=(2pi*period(hv.h.f)[1]), deriv=(raise_multiplier(deriv(hv.h.f)[1]),), offset=offset(hv.h.f), shift=shift(hv.h.f)), tpx)
    # compute second derivative of the current dimension
    m = map(y -> map(x -> evaluate(x, tpx), y), hv.m)
    vdd = map(vi -> evaluate(FourierSeries(vi.c, period=period(vi), deriv=(raise_multiplier(deriv(hv.h.f)[1]),), offset=offset(hv.h.f), shift=shift(hv.h.f)), tpx), hv.v)
    vd2 = evaluate(FourierSeries(hv.h.f.c, period=map(x->2pi*x, period(hv.h.f)), deriv=(raise_multiplier(raise_multiplier(deriv(hv.h.f)[1])),), offset=offset(hv.h.f), shift=shift(hv.h.f)), tpx)
    # wrapup
    m_ = tunroll((vd2, vdd...), m...)
    h, vs, ms = to_vcomp_gauge_mass(hv, evaluate(hv.h, x), (v1, v...), m_)
    # We want a compact representation of the Hessian, which is symmetric, however we can't
    # use LinearAlgebra.Symmetric because it is recursive
    # https://github.com/JuliaLang/julia/pull/25688
    # so we need a SSymmetricCompact type
    masses = SVector{StaticArrays.triangularnumber(length(vs)),eltype(ms)}(ms)
    return (h, to_coord(hv, hv.A, SVector(vs)), to_coord(hv, hv.A, SSymmetricCompact(masses)))
end

tunroll() = ()
tunroll(x::Tuple, y::Tuple...) = (x..., tunroll(y...)...)

function Base.zero(::Type{FourierValue{X,S}}) where {X,A,B,C,D<:Eigen{A,B,C},V,M,S<:Tuple{D,V,M}}
    return FourierValue(zero(X),(eigen(Hermitian(zero(C))), zero(V), zero(M)))
end

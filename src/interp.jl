"""
    HamiltonianInterp(f::FourierSeries; gauge=:Wannier)

A wrapper for `FourierSeries` with an additional gauge that allows for
convenient diagonalization of the result. For details see [`to_gauge`](@ref).
"""
struct HamiltonianInterp{G,S,N,iip,C,A,T,F} <: AbstractHamiltonianInterp{G,N,T,iip}
    f::FourierSeries{S,N,iip,C,A,T,F}
    function HamiltonianInterp{G}(f::FourierSeries{S,N,iip,C,A,T,F}) where {G,S,N,iip,C,A,T,F}
        return new{G,S,N,iip,C,A,T,F}(f)
    end
end

HamiltonianInterp(f; gauge=GaugeDefault(HamiltonianInterp)) = HamiltonianInterp{gauge}(f)

GaugeDefault(::Type{<:HamiltonianInterp}) = Wannier()
Base.parent(h::HamiltonianInterp) = h.f

period(h::HamiltonianInterp) = period(parent(h))
frequency(h::HamiltonianInterp) = frequency(parent(h))
allocate(h::HamiltonianInterp, x, dim) = allocate(h.f, x, dim)
function contract!(cache, h::HamiltonianInterp, x, dim)
    return HamiltonianInterp{gauge(h)}(contract!(cache, parent(h), x, dim))
end
function evaluate!(cache, h::HamiltonianInterp, x)
    return to_gauge(h, evaluate!(cache, parent(h), x))
end
nextderivative(h::HamiltonianInterp, dim) = nextderivative(parent(h), dim)

# ------------------------------------------------------------------------------

struct BerryConnectionInterp{P,B,G,N,T,iip,S,F,TB} <: AbstractCoordInterp{B,G,N,T,iip}
    a::ManyFourierSeries{N,T,iip,S,F}
    B::TB
    BerryConnectionInterp{P,B}(a::ManyFourierSeries{N,T,iip,S,F}, b::TB) where {P,B,N,T,iip,S,F,TB} =
        new{P,B,GaugeDefault(BerryConnectionInterp),N,T,iip,S,F,TB}(a, b)
end

function BerryConnectionInterp{P}(a, B; coord=CoordDefault(BerryConnectionInterp{P})) where P
    return BerryConnectionInterp{P,coord}(a, B)
end

period(bc::BerryConnectionInterp) = period(bc.a)
frequency(bc::BerryConnectionInterp) = frequency(bc.a)

GaugeDefault(::Type{<:BerryConnectionInterp}) = Wannier()
CoordDefault(::Type{<:BerryConnectionInterp}) = Cartesian()
CoordDefault(::Type{<:BerryConnectionInterp{P}}) where P = P

allocate(bc::BerryConnectionInterp, x, dim) = allocate(bc.a, x, dim)
function contract!(cache, bc::BerryConnectionInterp{P,B}, x, dim) where {P,B}
    a = contract!(cache, bc.a, x, dim)
    return BerryConnectionInterp{P,B}(a, bc.B)
end
function evaluate!(cache, bc::BerryConnectionInterp, x)
    as = map(herm, evaluate!(cache, bc.a, x)) # take Hermitian part of A since Wannier90 may not enforce it
    return to_coord(bc, bc.B, SVector(as))
end

# ------------------------------------------------------------------------------

struct GradientVelocityInterp{C,B,G,N,T,iip,F,DF,TA} <: AbstractVelocityInterp{C,B,G,N,T,iip}
    j::JacobianSeries{N,T,iip,F,DF}
    A::TA
    function GradientVelocityInterp{C,B,G}(j::JacobianSeries{N,T,iip,F,DF}, A::TA) where {C,B,G,N,T,iip,F,DF,TA}
        @assert gauge(parent(j)) == GaugeDefault(parent(j))
        return new{C,B,G,N,T,iip,F,DF,TA}(j, A)
    end
end

function GradientVelocityInterp(h::AbstractHamiltonianInterp, A;
    gauge=GaugeDefault(GradientVelocityInterp),
    coord=CoordDefault(GradientVelocityInterp),
    vcomp=VcompDefault(GradientVelocityInterp))
    return GradientVelocityInterp{vcomp,coord,gauge}(JacobianSeries(h), A)
end

Base.parent(hv::GradientVelocityInterp) = hv.j
CoordDefault(::Type{<:GradientVelocityInterp}) = Lattice()
GaugeDefault(::Type{<:GradientVelocityInterp}) = Wannier()
VcompDefault(::Type{<:GradientVelocityInterp}) = Whole()

period(hv::GradientVelocityInterp) = period(parent(hv))
frequency(hv::GradientVelocityInterp) = frequency(parent(hv))
allocate(hv::GradientVelocityInterp, x, dim) = allocate(parent(hv), x, dim)
function contract!(cache, hv::GradientVelocityInterp{C,B,G}, x, dim) where {C,B,G}
    return GradientVelocityInterp{C,B,G}(contract!(cache, parent(hv), x, dim), hv.A)
end
function evaluate!(cache, hv::GradientVelocityInterp, x)
    h, vs = to_vcomp_gauge(hv, evaluate!(cache, parent(hv), x)...)
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
struct CovariantVelocityInterp{C,B,G,N,T,iip,HV,A} <: AbstractVelocityInterp{C,B,G,N,T,iip}
    hv::HV
    a::A
    function CovariantVelocityInterp{C,B,G}(hv::HV, a::A) where {C,B,G,HV<:GradientVelocityInterp,A<:BerryConnectionInterp{B}}
        @assert ndims(hv) == ndims(a)
        @assert FourierSeriesEvaluators.isinplace(hv) == FourierSeriesEvaluators.isinplace(a)
        @assert vcomp(hv) == VcompDefault(HV)
        @assert gauge(hv) == GaugeDefault(HV)
        @assert coord(hv) == B
        iip = FourierSeriesEvaluators.isinplace(hv)
        return new{C,B,G,ndims(hv),eltype(period(hv)),iip,HV,A}(hv, a)
    end
end
function CovariantVelocityInterp(hv, a;
    gauge=GaugeDefault(CovariantVelocityInterp),
    coord=CoordDefault(CovariantVelocityInterp),
    vcomp=VcompDefault(CovariantVelocityInterp))
    @assert ndims(hv) == length(a.a.s)
    @assert AutoBZ.coord(hv) == AutoBZ.coord(a) # function and keyword name conflict
    # TODO convert hv and A to the requested coordinate instead of throwing
    # method error in inner constructor
    return CovariantVelocityInterp{vcomp,coord,gauge}(hv, a)
end

CoordDefault(::Type{<:CovariantVelocityInterp}) = Lattice()
GaugeDefault(::Type{<:CovariantVelocityInterp}) = Wannier()
VcompDefault(::Type{<:CovariantVelocityInterp}) = Whole()

Base.parent(chv::CovariantVelocityInterp) = parent(chv.hv)
period(chv::CovariantVelocityInterp) = period(parent(chv))
frequency(chv::CovariantVelocityInterp) = frequency(parent(chv))
function allocate(chv::CovariantVelocityInterp, x, dim)
    hv_cache = allocate(chv.hv, x, dim)
    a_cache = allocate(chv.a, x * period(chv.a, dim) / period(chv, dim), dim)
    return (hv_cache, a_cache)
end
function contract!(cache, chv::CovariantVelocityInterp{C,B,G}, x, dim) where {C,B,G}
    hv = contract!(cache[1], chv.hv, x, dim)
    a = contract!(cache[2], chv.a, x * period(chv.a, dim) / period(chv, dim), dim)
    return CovariantVelocityInterp{C,B,G}(hv, a)
end
function evaluate!(cache, chv::CovariantVelocityInterp, x)
    hw, vws = evaluate!(cache[1], chv.hv, x)
    as = evaluate!(cache[2], chv.a, x * period(chv.a, 1) * frequency(chv, 1))
    return to_vcomp_gauge(chv, hw, map((v, a) -> covariant_velocity(hw, v, a), vws, as))
end

# some special methods for inferring the rule
@inline zero_eig(::Type{T}) where {T} = zero(T)
@inline zero_eig(::Type{<:Eigen{A,B,C}}) where {A,B,C} = eigen(Hermitian(zero(C)))

# for HamiltonianInterp
function Base.zero(::Type{FourierValue{X,S}}) where {X,S<:Eigen}
    return FourierValue(zero(X),zero_eig(S))
end
# for the velocity interps
function Base.zero(::Type{FourierValue{X,T}}) where {X,T<:Tuple}
    return FourierValue(zero(X), ntuple(i -> zero_eig(fieldtype(T,i)),Val(fieldcount(T))))
end
# ----------------


struct MassVelocityInterp{C,B,G,N,T,iip,F,DF,TA} <: AbstractVelocityInterp{C,B,G,N,T,iip}
    h::HessianSeries{N,T,iip,F,DF}
    A::TA
    function MassVelocityInterp{C,B,G}(h::HessianSeries{N,T,iip,F,DF}, A::TA) where {C,B,G,N,T,iip,F,DF,TA}
        @assert gauge(parent(h)) == GaugeDefault(parent(h))
        return new{C,B,G,N,T,iip,F,DF,TA}(h, A)
    end
end

function MassVelocityInterp(h::AbstractHamiltonianInterp, A;
    gauge=GaugeDefault(MassVelocityInterp),
    coord=CoordDefault(MassVelocityInterp),
    vcomp=VcompDefault(MassVelocityInterp))
    MassVelocityInterp{vcomp,coord,gauge}(HessianSeries(h), A)
end

CoordDefault(::Type{<:MassVelocityInterp}) = Lattice()
GaugeDefault(::Type{<:MassVelocityInterp}) = Wannier()
VcompDefault(::Type{<:MassVelocityInterp}) = Whole()
Base.parent(mv::MassVelocityInterp) = mv.h
period(mv::MassVelocityInterp) = period(parent(mv))
frequency(mv::MassVelocityInterp) = frequency(parent(mv))
allocate(mv::MassVelocityInterp, x, dim) = allocate(parent(mv), x, dim)
function contract!(cache, mv::MassVelocityInterp{C,B,G}, x, dim) where {C,B,G}
    h = contract!(cache, parent(mv), x, dim)
    return MassVelocityInterp{C,B,G}(h, mv.A)
end
function evaluate!(cache, mv::MassVelocityInterp, x)
    hw, vws, mws = evaluate!(cache, parent(mv), x)
    h, vs, ms = to_vcomp_gauge_mass(mv, hw, vws, tunroll(mws...))
    # We want a compact representation of the Hessian, which is symmetric, however we can't
    # use LinearAlgebra.Symmetric because it is recursive
    # https://github.com/JuliaLang/julia/pull/25688
    # so we need a SSymmetricCompact type
    masses = SVector{StaticArrays.triangularnumber(length(vs)),eltype(ms)}(ms)
    return (h, to_coord(mv, mv.A, SVector(vs)), to_coord(mv, mv.A, SSymmetricCompact(masses)))
end

tunroll() = ()
tunroll(x::Tuple, y::Tuple...) = (x..., tunroll(y...)...)

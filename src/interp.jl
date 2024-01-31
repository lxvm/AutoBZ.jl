# this wrapper houses a FourierSeries in radians but evaluates it in the frequency variable
# i.e. if s is 2π-periodic, then Freq2RadSeries is 1-periodic
# NOTE that the derivatives of this series are not rescaled by 2π, intentionally!
# this is needed for evaluating gradients of operators with the appropriate angular convention
struct Freq2RadSeries{N,T,iip,S<:AbstractFourierSeries{N,T,iip},Tt<:NTuple{N,T},Tf<:NTuple{N,Any}} <: AbstractFourierSeries{N,T,iip}
    s::S
    t::Tt
    f::Tf
end

function Freq2RadSeries(s::AbstractFourierSeries)
    f = map(freq2rad, frequency(s))
    t = map(inv, f)
    return Freq2RadSeries(s, t, f)
end

period(s::Freq2RadSeries) = s.t
frequency(s::Freq2RadSeries) = s.f
allocate(s::Freq2RadSeries, x, dim) = allocate(s.s, freq2rad(x), dim)
function contract!(cache, s::Freq2RadSeries, x, dim)
    t = FourierSeriesEvaluators.deleteat_(s.t, dim)
    f = FourierSeriesEvaluators.deleteat_(s.f, dim)
    return Freq2RadSeries(contract!(cache, s.s, freq2rad(x), dim), t, f)
end
function evaluate!(cache, s::Freq2RadSeries, x)
    return evaluate!(cache, s.s, freq2rad(x))
end
function nextderivative(s::Freq2RadSeries, dim)
    return Freq2RadSeries(nextderivative(s.s, dim), s.t, s.f)
end

show_dims(s::Freq2RadSeries) = show_dims(s.s)
show_details(s::Freq2RadSeries) = show_details(s.s)


"""
    HamiltonianInterp(f::FourierSeries; gauge=:Wannier)

A wrapper for `FourierSeries` with an additional gauge that allows for
convenient diagonalization of the result. For details see [`to_gauge`](@ref).
"""
struct HamiltonianInterp{G,N,T,iip,F<:Freq2RadSeries{N,T,iip,<:FourierSeries}} <: AbstractHamiltonianInterp{G,N,T,iip}
    f::F
    function HamiltonianInterp{G}(f::Freq2RadSeries{N,T,iip,<:FourierSeries}) where {G,N,T,iip}
        return new{G,N,T,iip,typeof(f)}(f)
    end
end

HamiltonianInterp(f; gauge=GaugeDefault(HamiltonianInterp)) = HamiltonianInterp{gauge}(f)

GaugeDefault(::Type{<:HamiltonianInterp}) = Wannier()
parentseries(h::HamiltonianInterp) = h.f.s

period(h::HamiltonianInterp) = period(h.f)
frequency(h::HamiltonianInterp) = frequency(h.f)
allocate(h::HamiltonianInterp, x, dim) = allocate(h.f, x, dim)
function contract!(cache, h::HamiltonianInterp, x, dim)
    return HamiltonianInterp{gauge(h)}(contract!(cache, h.f, x, dim))
end
function evaluate!(cache, h::HamiltonianInterp, x)
    return to_gauge(h, evaluate!(cache, h.f, x))
end
nextderivative(h::HamiltonianInterp, dim) = nextderivative(h.f, dim)

# ------------------------------------------------------------------------------

"""
    BerryConnectionInterp{P}(a::ManyFourierSeries, B; coord)

Interpolate the Berry connection in basis `coord`. `a` must evaluate the components of the
connection in coordinate basis `P`, and `B` is the coordinate transformation from `P` to
`coord`.
"""
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

_parentseries(d::DerivativeSeries{1}) = d.f
_parentseries(d::DerivativeSeries) = _parentseries(d.f)

"""
    GradientVelocityInterp(h::AbstractHamiltonianInterp, A; gauge, coord, vcomp)

Evaluate the Hamiltonian and its gradient, which doesn't produce gauge-covariant velocities.
The Hamiltonian `h` must be in the Wannier gauge, but this will give the result in the
requested `gauge`. `A` must be the coordinate transformation from the lattice basis to the
desired `coord` system. `vcomp` selects the contribution to the velocities.
"""
struct GradientVelocityInterp{C,B,G,N,T,iip,F,DF,TA} <: AbstractVelocityInterp{C,B,G,N,T,iip}
    j::JacobianSeries{N,T,iip,F,DF}
    A::TA
    function GradientVelocityInterp{C,B,G}(j::JacobianSeries{N,T,iip,F,DF}, A::TA) where {C,B,G,N,T,iip,F,DF,TA}
        @assert gauge(_parentseries(j)) == GaugeDefault(_parentseries(j))
        return new{C,B,G,N,T,iip,F,DF,TA}(j, A)
    end
end

function GradientVelocityInterp(h::AbstractHamiltonianInterp, A;
    gauge=GaugeDefault(GradientVelocityInterp),
    coord=CoordDefault(GradientVelocityInterp),
    vcomp=VcompDefault(GradientVelocityInterp))
    return GradientVelocityInterp{vcomp,coord,gauge}(JacobianSeries(h), A)
end

parentseries(hv::GradientVelocityInterp) = _parentseries(hv.j)
CoordDefault(::Type{<:GradientVelocityInterp}) = Lattice()
GaugeDefault(::Type{<:GradientVelocityInterp}) = Wannier()
VcompDefault(::Type{<:GradientVelocityInterp}) = Whole()

period(hv::GradientVelocityInterp) = period(hv.j)
frequency(hv::GradientVelocityInterp) = frequency(hv.j)
allocate(hv::GradientVelocityInterp, x, dim) = allocate(hv.j, x, dim)
function contract!(cache, hv::GradientVelocityInterp{C,B,G}, x, dim) where {C,B,G}
    return GradientVelocityInterp{C,B,G}(contract!(cache, hv.j, x, dim), hv.A)
end
function evaluate!(cache, hv::GradientVelocityInterp, x)
    h, vs = to_vcomp_gauge(hv, evaluate!(cache, hv.j, x)...)
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
    CovariantVelocityInterp(hv::GradientVelocityInterp, a::BerryConnectionInterp)

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

parentseries(chv::CovariantVelocityInterp) = parentseries(chv.hv)
period(chv::CovariantVelocityInterp) = period(chv.hv)
frequency(chv::CovariantVelocityInterp) = frequency(chv.hv)
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
# function Base.zero(::Type{FourierValue{X,S}}) where {X,S<:Eigen}
#     return FourierValue(zero(X),zero_eig(S))
# end
# for the velocity interps
# function Base.zero(::Type{FourierValue{X,T}}) where {X,T<:Tuple}
#     return FourierValue(zero(X), ntuple(i -> zero_eig(fieldtype(T,i)),Val(fieldcount(T))))
# end
# ----------------

"""
    MassVelocityInterp(h::AbstractHamiltonianInterp, A; gauge, coord, vcomp)

Compute the Hamiltonian, its gradient and Hessian, which are not gauge-covariant. See
[`GradientVelocityInterp`](@ref) for explanation of the arguments
"""
struct MassVelocityInterp{C,B,G,N,T,iip,F,DF,TA} <: AbstractVelocityInterp{C,B,G,N,T,iip}
    h::HessianSeries{N,T,iip,F,DF}
    A::TA
    function MassVelocityInterp{C,B,G}(h::HessianSeries{N,T,iip,F,DF}, A::TA) where {C,B,G,N,T,iip,F,DF,TA}
        @assert gauge(_parentseries(h)) == GaugeDefault(_parentseries(h))
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
parentseries(mv::MassVelocityInterp) = _parentseries(mv.h)
period(mv::MassVelocityInterp) = period(mv.h)
frequency(mv::MassVelocityInterp) = frequency(mv.h)
allocate(mv::MassVelocityInterp, x, dim) = allocate(mv.h, x, dim)
function contract!(cache, mv::MassVelocityInterp{C,B,G}, x, dim) where {C,B,G}
    h = contract!(cache, mv.h, x, dim)
    return MassVelocityInterp{C,B,G}(h, mv.A)
end
function evaluate!(cache, mv::MassVelocityInterp, x)
    hw, vws, mws = evaluate!(cache, mv.h, x)
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

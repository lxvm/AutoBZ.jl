"""
    HamiltonianInterp(f::InplaceFourierSeries; gauge=:Wannier)

A wrapper for `InplaceFourierSeries` with an additional gauge that allows for
convenient diagonalization of the result. For details see [`to_gauge`](@ref).
"""
struct HamiltonianInterp{G,N,T,F} <: AbstractGaugeInterp{G,N,T}
    f::F
    HamiltonianInterp{G}(f::F) where {G,F<:Union{FourierSeries,InplaceFourierSeries}} =
        new{G,ndims(f),eltype(f),F}(f)
end

# recursively wrap inner Fourier series with Hamiltonian
HamiltonianInterp(f; gauge=GaugeDefault(HamiltonianInterp)) =
    HamiltonianInterp{gauge}(f)

GaugeDefault(::Type{<:HamiltonianInterp}) = Wannier()


"""
    shift!(h::HamiltonianInterp, λ::Number)

Modifies and returns `h` such that it returns `h - λ*I`. Will throw a
`BoundsError` if this operation cannot be done on the existing data.
"""
function shift!(h::HamiltonianInterp, λ_::Number)
    λ = convert(eltype(eltype(h)), λ_)
    c = coefficients(h)
    idx = first(CartesianIndices(c)).I .- offset(h.f) .- 1
    h.f.c[idx...] -= λ*I
    return h
end

period(h::HamiltonianInterp) = period(h.f)

contract(h::HamiltonianInterp, x::Number, ::Val{d}) where d =
    HamiltonianInterp{gauge(h)}(contract(h.f, x, Val(d)))

evaluate(h::HamiltonianInterp, x::NTuple{1}) =
    to_gauge(h, evaluate(h.f, x))

coefficients(h::HamiltonianInterp) = coefficients(h.f)

function fourier_type(h::HamiltonianInterp, x)
    T = fourier_type(h.f, x)
    if gauge(h) isa Wannier
        return T
    else
        return typeof(eigen(Hermitian(one(T))))
    end
end

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
CoordDefault(::Type{<:BerryConnectionInterp{P}}) where P = P

contract(bc::BerryConnectionInterp, x::Number, ::Val{d}) where d =
    BerryConnectionInterp{CoordDefault(bc),coord(bc)}(contract(bc.a, x, Val(d)), bc.B)

function evaluate(bc::BerryConnectionInterp, x::NTuple{1})
    as = map(herm, evaluate(bc.a, x)) # take Hermitian part of A since Wannier90 may not enforce it
    return to_coord(bc, bc.B, SVector(as))
end

function fourier_type(a::BerryConnectionInterp, x)
    T = fourier_type(eltype(eltype(a.a)), x)
    return SVector{length(a.a.fs),T}
end

# ------------------------------------------------------------------------------

# These methods are missing from FourierSeriesEvaluators.jl
deriv(f::FourierSeries) = f.a
offset(f::FourierSeries) = f.o
shift(f::FourierSeries) = f.q
function contract(f::FourierSeries{N,T}, x::Number, ::Val{N}) where {N,T}
    c = fourier_contract(f.c, x-f.q[N], f.k[N], f.a[N], f.o[N], Val(N))
    k = Base.front(f.k)
    a = Base.front(f.a)
    o = Base.front(f.o)
    q = Base.front(f.q)
    FourierSeries{N-1,eltype(c)}(c, k, a, o, q)
end

raise_multiplier(::Val{0}) = Val(1)
raise_multiplier(::Val{1}) = 2
raise_multiplier(a) = a + 1


"""
    GradientVelocityInterp(H::Hamiltonian{Val(:Wannier)}, A; gauge=:Wannier, vcord=:lattice, vcomp=:whole)

Evaluates the band velocities by directly computing the Hamiltonian gradient,
which is not gauge-covariant. Returns a tuple of the Hamiltonian and the three
velocity matrices. See [`to_vcomp_gauge`](@ref) for the `vcomp` keyword.
`A` should be the matrix of lattice vectors, which is used only if `vcord` is
`:cartesian`.
"""
struct GradientVelocityInterp{C,B,G,N,T,H,U,V,TA} <: AbstractVelocityInterp{C,B,G,N,T}
    h::H
    u::U
    v::V
    A::TA
    GradientVelocityInterp{C,B,G}(h::H, u::U, v::V, A::TA) where {C,B,G,H<:HamiltonianInterp{GaugeDefault(HamiltonianInterp)},U,V,TA} =
        new{C,B,G,ndims(h),eltype(h),H,U,V,TA}(h, u, v, A)
end

function GradientVelocityInterp(h::HamiltonianInterp{GaugeDefault(HamiltonianInterp)}, A;
    gauge=GaugeDefault(GradientVelocityInterp),
    coord=CoordDefault(GradientVelocityInterp),
    vcomp=VcompDefault(GradientVelocityInterp))
    T = h.f isa InplaceFourierSeries ? InplaceFourierSeries : FourierSeries
    u = (); cd = coefficients(h); d = ndims(h)
    while d > 0
        v = view(cd, ntuple(n -> n>d ? first(axes(cd,n)) : axes(cd,n), Val(ndims(h)))...)
        c = similar(v, fourier_type(eltype(h),eltype(period(h.f))))
        ud = T(c; period=2pi .* period(h.f)[1:d], deriv=(deriv(h.f)[1:d-1]..., raise_multiplier(deriv(h.f)[d])), offset=offset(h.f)[1:d], shift=shift(h.f)[1:d])
        u = (ud, u...)
        d -= 1
    end
    u[ndims(h)].c .= h.f.c
    GradientVelocityInterp{vcomp,coord,gauge}(h, u[1:ndims(h)-1], (u[ndims(h)],), A)
end

hamiltonian(hv::GradientVelocityInterp) = hv.h
CoordDefault(::Type{<:GradientVelocityInterp}) = Lattice()
GaugeDefault(::Type{<:GradientVelocityInterp}) = Wannier()
VcompDefault(::Type{<:GradientVelocityInterp}) = Whole()

function contract(hv::GradientVelocityInterp{C,B,G}, x::Number, ::Val{d}) where {C,B,G,d}
    h = contract(hv.h, x, Val(d))
    tpx = 2pi*x
    v = map(vi -> contract(vi, tpx, Val(d)), hv.v)
    u = hv.u[1:d-2]
    vd = hv.u[d-1]
    vd.c .= h.f.c # copy contracted coefficients to velocity
    GradientVelocityInterp{C,B,G}(h, u, (vd, v...), hv.A)
end

function evaluate(hv::GradientVelocityInterp, x::NTuple{1})
    tpx = (2pi*x[1],)   # period adjusted to correct for angular coordinates in derivative
    h, vs = to_vcomp_gauge(hv, evaluate(hv.h, x),  map(v -> evaluate(v, tpx), hv.v))
    return (h, to_coord(hv, hv.A, SVector(vs)))
end

function fourier_type(hv::GradientVelocityInterp, x)
    T = fourier_type(hv.h, x)
    V = SVector{length(hv.u)+length(hv.v),T}
    if gauge(hv) isa Wannier
        return Tuple{T,V}
    else
        return Tuple{typeof(eigen(Hermitian(one(T)))),V}
    end
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
    CovariantVelocityInterp{C,B,G}(hv::HV, a::A) where {C,B,G,HV<:GradientVelocityInterp{VcompDefault(GradientVelocityInterp),B,GaugeDefault(GradientVelocityInterp)},A<:BerryConnectionInterp{B}} =
        new{C,B,G,ndims(hv),eltype(hv),HV,A}(hv, a)
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

function fourier_type(hv::CovariantVelocityInterp, x)
    T = fourier_type(hv.hv.h, x)
    V = fourier_type(hv.a, x)
    if gauge(hv) isa Wannier
        return Tuple{T,V}
    else
        return Tuple{typeof(eigen(Hermitian(one(T)))),V}
    end
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

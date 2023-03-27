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
CoordDefault(::Type{<:CovariantVelocityInterp}) = Cartesian()
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


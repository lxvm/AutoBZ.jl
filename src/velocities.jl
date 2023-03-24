raise_multiplier(::Val{0}) = Val(1)
raise_multiplier(::Val{1}) = 2
raise_multiplier(a) = a + 1

"""
    to_vcomp_gauge(::Val{C}, ::Val{G}, h, vs...) where {C,G}

Take the velocity components of `vs` in any gauge according to the value of `C`
- `:whole`: return the whole velocity (sum of interband and intraband components)
- `:intra`: return the intraband velocity (diagonal in Hamiltonian gauge)
- `:inter`: return the interband velocity (off-diagonal terms in Hamiltonian gauge)
    
Transform the velocities into a gauge according to the following values of `G`
- `:Wannier`: keeps `H, vs` in the original, orbital basis
- `:Hamiltonian`: diagonalizes `H` and rotates `H, vs` into the energy, band basis
"""
to_vcomp_gauge(vcomp::C, gauge::G, H, vs::AbstractMatrix...) where {C,G} =
    to_vcomp_gauge(vcomp, gauge, H, vs)

to_vcomp_gauge(::Val{:whole}, ::Val{:Wannier}, H, vs::NTuple) = (H, vs...)
function to_vcomp_gauge(vcomp::C, w::Val{:Wannier}, H, vs::NTuple) where C
    E, vhs... = to_vcomp_gauge(vcomp, Val(:Hamiltonian), H, vs)
    e, vws = to_gauge(w, E, vhs)
    (e, vws...)
end

function to_vcomp_gauge(vcomp::C, ::Val{:Hamiltonian}, H::AbstractMatrix, vws::NTuple{N}) where {C,N}
    E, vhs = to_gauge(Val(:Hamiltonian), H, vws)
    (E, to_vcomp(vcomp, vhs)...)
end

function to_gauge(::Val{:Wannier}, H::Eigen, vhs::NTuple{N}) where N
    U = H.vectors
    to_gauge(Val(:Wannier), H), ntuple(n -> U * vhs[n] * U', Val(N))
end
function to_gauge(::Val{:Hamiltonian}, H::AbstractMatrix, vws::NTuple{N}) where N
    E = to_gauge(Val(:Hamiltonian), H)
    U = E.vectors
    E, ntuple(n -> U' * vws[n] * U, Val(N))
end

to_vcomp(::Val{:whole}, vhs::NTuple{N,T}) where {N,T} = vhs
to_vcomp(::Val{:inter}, vhs::NTuple{N,T}) where {N,T} =
    ntuple(n -> vhs[n] - Diagonal(vhs[n]), Val(N))
to_vcomp(::Val{:intra}, vhs::NTuple{N,T}) where {N,T} =
    ntuple(n -> Diagonal(vhs[n]), Val(N))

"""
    HamiltonianVelocity(H::Hamiltonian{Val(:Wannier)}, A; gauge=:Wannier, vcord=:lattice, vcomp=:whole)

Evaluates the band velocities by directly computing the Hamiltonian gradient,
which is not gauge-covariant. Returns a tuple of the Hamiltonian and the three
velocity matrices. See [`to_vcomp_gauge`](@ref) for the `vcomp` keyword.
`A` should be the matrix of lattice vectors, which is used only if `vcord` is
`:cartesian`.
"""
struct HamiltonianVelocity{B,C,G,N,T,H,U,V,TA} <: AbstractVelocity{B,C,G,N,T}
    h::H
    u::U
    v::V
    A::TA
    HamiltonianVelocity{B,C,G}(h::H, u::U, v::V, A::TA) where {B,C,G,H<:Hamiltonian{Val{:Wannier}()},U,V,TA} =
        new{B,C,G,ndims(h),eltype(h),H,U,V,TA}(h, u, v, A)
end

function HamiltonianVelocity(h::Hamiltonian{Val(:Wannier)}, A; gauge=:Wannier, vcord=:lattice, vcomp=:whole)
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
    HamiltonianVelocity{Val(vcord),Val(vcomp),Val(gauge)}(h, u[1:ndims(h)-1], (u[ndims(h)],), A)
end

hamiltonian(hv::HamiltonianVelocity) = hamiltonian(hv.h)

function contract(hv::HamiltonianVelocity{B,C,G}, x::Number, ::Val{d}) where {B,C,G,d}
    h = contract(hv.h, x, Val(d))
    tpx = 2pi*x
    v = map(vi -> contract(vi, tpx, Val(d)), hv.v)
    u = hv.u[1:d-2]
    vd = hv.u[d-1]
    vd.c .= h.f.c # copy contracted coefficients to velocity
    HamiltonianVelocity{B,C,G}(h, u, (vd, v...), hv.A)
end

"""
    to_vcord(B, A, vs)

If `B::Val{:lattice}` return `vs` and if `B::Val{:cartesian}` return `A*vs`
"""
to_vcord(::Val{:lattice}, A, vs) = vs
to_vcord(::Val{:cartesian}, A, vs) = A * vs
to_vcord(::Val, A, vs) = throw(ArgumentError("vcord $(vcord(hv)) unrecognized. use `:lattice` or `:cartesian`"))


function evaluate(hv::HamiltonianVelocity{B,C,G,1}, x::NTuple{1}) where {B,C,G}
    tpx = (2pi*x[1],)   # period adjusted to correct for angular coordinates in derivative
    h, vs... = to_vcomp_gauge(C, G, evaluate(hv.h, x),  map(v -> evaluate(v, tpx), hv.v)...)
    return (h, to_vcord(B, hv.A, SVector(vs)))
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
    CovariantHamiltonianVelocity(hv::HamiltonianVelocity{Val(:Wannier)}, a::ManyFourierSeries)

Uses the Berry connection to return fully gauge-covariant velocities. Returns a
tuple of the Hamiltonian and the three velocity matrices.
"""
struct CovariantHamiltonianVelocity{B,C,G,N,T,HV,A} <: AbstractVelocity{B,C,G,N,T}
    hv::HV
    a::A
    CovariantHamiltonianVelocity{B,C,G}(hv::HV, a::A) where {B,C,G,HV<:HamiltonianVelocity{Val{:lattice}(),Val{:whole}(),Val{:Wannier}()},A} =
        new{B,C,G,ndims(hv),eltype(hv),HV,A}(hv, a)
end
function CovariantHamiltonianVelocity(hv::HV, a::A; gauge=:Wannier, vcord=:lattice, vcomp=:whole) where {HV<:HamiltonianVelocity,A<:ManyFourierSeries}
    @assert period(hv) == period(a)
    @assert ndims(hv) == length(a.fs)
    CovariantHamiltonianVelocity{Val(vcord),Val(vcomp),Val(gauge)}(hv, a)
end

hamiltonian(chv::CovariantHamiltonianVelocity) = hamiltonian(chv.hv)

contract(chv::CovariantHamiltonianVelocity{B,C,G}, x::Number, ::Val{d}) where {B,C,G,d} =
    CovariantHamiltonianVelocity{B,C,G}(contract(chv.hv, x, Val(d)), contract(chv.a, x, Val(d)))

function evaluate(chv::CovariantHamiltonianVelocity{B,C,G,1}, x::NTuple{1}) where {B,C,G}
    hw, vws = evaluate(chv.hv, x)
    as = evaluate(chv.a, x)
    h, vs... = to_vcomp_gauge(C, G, hw, map((v, a) -> covariant_velocity(hw, v, a), vws.data, as))
    return (h, to_vcord(B, chv.hv.A, SVector(vs)))
end


"""
    HamiltonianVelocity(H::Hamiltonian{Val(:Wannier)}; vcomp=:whole)

Evaluates the band velocities by directly computing the Hamiltonian gradient,
which is not gauge-covariant. Returns a tuple of the Hamiltonian and the three
velocity matrices. See [`to_vcomp_gauge`](@ref) for the `vcomp` keyword.
"""
struct HamiltonianVelocity{C,G,N,T,HV,H,V} <: AbstractVelocity{C,G,N,T}
    hv::HV
    h::H
    v::V
    HamiltonianVelocity{C,G}(hv::HV, h::H, v::V) where {C,G,HV,H<:Hamiltonian,V} =
        new{C,G,ndims(h),eltype(h),HV,H,V}(hv, h, v)
end

function HamiltonianVelocity(h::Hamiltonian{Val(:Wannier)}, v=(); gauge=:Wannier, vcomp=:whole)
    f = InplaceFourierSeries(similar(h.h.f.c); period=period(h.h.f), deriv=h.h.f.a, offset=h.h.f.o, shift=h.h.f.q)
    hv = HamiltonianVelocity(h.h, (f, map(u -> u.f, v)...); gauge=gauge, vcomp=vcomp)
    HamiltonianVelocity{Val(vcomp),Val(gauge)}(hv, h, v)
end
HamiltonianVelocity(h::Hamiltonian{Val(:Wannier),0}, v=(); gauge=:Wannier, vcomp=:whole) =
    HamiltonianVelocity{Val(vcomp),Val(gauge)}((), h, v)

hamiltonian(hv::HamiltonianVelocity) = hamiltonian(hv.h)

function contract!(hv::HamiltonianVelocity, x::Number, ::Val{d}) where d
    contract!(hv.h, x, Val(d))
    for v in hv.v
        contract!(v, x, Val(d))
    end
    f = hv.h.f
    fourier_contract!(coefficients(hv.hv.v[1]), coefficients(hv), x-f.q, f.k, raise_multiplier(f.a), f.o, Val(d))
    return hv.hv
end

raise_multiplier(::Val{0}) = Val(1)
raise_multiplier(::Val{1}) = 2
raise_multiplier(a) = a + 1

function evaluate(hv::HamiltonianVelocity{C,G,1}, x::NTuple{1}) where {C,G}
    f = hv.h.f
    v1 = fourier_evaluate(coefficients(hv), x[1]-f.q, f.k, raise_multiplier(f.a), f.o)
    to_vcomp_gauge(C, G, evaluate(hv.h, x), v1, map(v -> evaluate(v, x), hv.v)...)
end

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
    E, vhs... = to_vcomp_gauge(vcomp, Val{:Hamiltonian}(), H, vs)
    to_gauge(w, E, vhs)
end

to_vcomp_gauge(::Val{:Hamiltonian}, ::Val{:whole}, H, vws::NTuple) =
    to_hamiltonian_gauge(H, vws)

function to_vcomp_gauge(::Val{:Hamiltonian}, ::Val{:inter}, H, vws::NTuple{N,T}) where {N,T}
    ϵ, vhs... = to_hamiltonian_gauge(H, vws)
    (ϵ, ntuple(n -> vhs[n] - Diagonal(vhs[n]), Val{N}())...)
end

function to_vcomp_gauge(::Val{:Hamiltonian}, ::Val{:intra}, H, vws::NTuple{N}) where N
    ϵ, vhs... = to_hamiltonian_gauge(H, vws)
    return (ϵ, ntuple(n -> Diagonal(vhs[n]), Val{N}())...)
end

function to_vcomp_gauge_type(::G, ::V, T) where {G,V}
    Base.promote_op(to_vcomp_gauge, G, V, T)
end

function to_gauge(::Val{:Wannier}, H::Eigen, vhs::NTuple{N}) where N
end
function to_gauge(::Val{:Hamiltonian}, H::AbstractMatrix, vws::NTuple{N}) where N
    ishermitian(H) || throw(ArgumentError("found non-Hermitian Hamiltonian"))
    eigen(Hermitian(H)) # need to wrap with Hermitian for type stability
end


"""
    CovariantHamiltonianVelocity(hv::HamiltonianVelocity{Val(:Wannier)}, a::ManyFourierSeries)

Uses the Berry connection to return fully gauge-covariant velocities. Returns a
tuple of the Hamiltonian and the three velocity matrices.
"""
struct CovariantHamiltonianVelocity{C,G,N,T,HV,A} <: AbstractVelocity{C,G,N,T}
    hv::HV
    a::A
    CovariantHamiltonianVelocity{C,G}(hv::HV, a::A) where {C,G,HV<:HamiltonianVelocity,A} =
        new{C,G,ndims(hv),eltype(hv),HV,A}(hv, a)
end
function CovariantHamiltonianVelocity(hv::HV, a::A; gauge=:Wannier, vcomp=:whole) where {HV<:HamiltonianVelocity,A<:ManyFourierSeries}
    @assert period(hv) == period(a)
    @assert ndims(hv) == length(a.fs)
    CovariantHamiltonianVelocity{Val(vcomp),Val(gauge)}(hv, a)
end

hamiltonian(chv::CovariantHamiltonianVelocity) = hamiltonian(chv.hv)

contract!(chv::CovariantHamiltonianVelocity{C,G}, x::Number, ::Val{d}) where {C,G,d} =
    CovariantHamiltonianVelocity{C,G}(contract!(chv.hv, x, Val(d)), contract(chv.a, x, Val(d)))

function evaluate(chv::CovariantHamiltonianVelocity{C,G,1}, x::NTuple{1}) where {C,G}
    h, vws... = evaluate(chv.hv, x)
    as = evaluate(chv.a, x)
    to_vcomp_gauge(C, G, h, map((v, a) -> covariant_velocity(h, v, a), vws, as))
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

export AbstractHamiltonianVelocity, vcomp, hamiltonian
export HamiltonianVelocity3D, CovariantHamiltonianVelocity3D

"""
    velocity(H, Hα, Aα)

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
velocity(H, ∂H_∂α, ∂A_∂α) = ∂H_∂α + (im*I)*commutator(H, ∂A_∂α)

"""
    band_velocities(::Type{Val{gauge}}, ::Type{Val{vcomp}}, H, vs...) where {gauge, kind}

Transform the band velocities according to the following values of `gauge`
- `:Wannier`: keeps `H, vs` in the original, orbital basis
- `:Hamiltonian`: diagonalizes `H` and rotates `H, vs` into the energy, band basis

Take the velocity components of `vs` in either gauge according to value of `vcomp`
- `:whole`: return the whole velocity (sum of interband and intraband components)
- `:intra`: return the intraband velocity (diagonal in Hamiltonian gauge)
- `:inter`: return the interband velocity (off-diagonal terms in Hamiltonian gauge)
"""
band_velocities(gauge::G, vcomp::V, H, vs::AbstractMatrix...) where {G,V} =
    band_velocities(gauge, vcomp, H, vs)

band_velocities(::Val{:Wannier}, ::Val{:whole}, H, vs::NTuple) = (H, vs...)
function band_velocities(::Val{:Wannier}, vcomp::V, H, vs::NTuple) where V
    E, vhs... = band_velocities(Val{:Hamiltonian}(), vcomp, H, vs)
    to_wannier_gauge(E, vhs)
end

band_velocities(::Val{:Hamiltonian}, ::Val{:whole}, H, vws::NTuple) =
    to_hamiltonian_gauge(H, vws)

function band_velocities(::Val{:Hamiltonian}, ::Val{:inter}, H, vws::NTuple{N,T}) where {N,T}
    ϵ, vhs... = to_hamiltonian_gauge(H, vws)
    (ϵ, ntuple(n -> vhs[n] - Diagonal(vhs[n]), Val{N}())...)
end

function band_velocities(::Val{:Hamiltonian}, ::Val{:intra}, H, vws::NTuple{N}) where N
    ϵ, vhs... = to_hamiltonian_gauge(H, vws)
    return (ϵ, ntuple(n -> Diagonal(vhs[n]), Val{N}())...)
end

function band_velocities_type(::G, ::V, T) where {G,V}
    Base.promote_op(band_velocities, G, V, T)
end

function to_gauge(::Val{:Wannier}, H::Eigen, vhs::NTuple{N}) where N
end
function to_gauge(::Val{:Hamiltonian}, H::AbstractMatrix, vws::NTuple{N}) where N
    ishermitian(H) || throw(ArgumentError("found non-Hermitian Hamiltonian"))
    eigen(Hermitian(H)) # need to wrap with Hermitian for type stability
end

"""
    AbstractHamiltonianVelocity{vcomp,gauge,N} <:AbstractWannierInterp{gauge,N}

An abstract substype of `AbstractWannierInterp` 
"""
abstract type AbstractHamiltonianVelocity{vcomp,gauge,N,T} <:AbstractWannierInterp{gauge,N,T} end
vcomp(::AbstractHamiltonianVelocity{vcomp}) where vcomp = vcomp

"""
    HamiltonianVelocity3D(coeffs; period=(1.0,1.0,1.0), kwargs...)
    HamiltonianVelocity3D(H::Hamiltonian3D{gauge}; vcomp=:whole) where gauge

Evaluates the band velocities by directly computing the Hamiltonian gradient,
which is not gauge-covariant. Returns a tuple of the Hamiltonian and the three
velocity matrices. See [`band_velocities`](@ref) for the `vcomp` keyword.
"""
struct HamiltonianVelocity3D{vcomp,gauge,T,TV,TH} <: AbstractHamiltonianVelocity{vcomp,gauge,3,T}
    H::Hamiltonian{gauge,T,TH}
    # vi_j where i represents array dims and j the coordinate index
    v2_3::Array{T,2}
    v1_3::Array{T,1}
    v1_2::Array{T,1}
    v0_3::Array{TV,0}
    v0_2::Array{TV,0}
    v0_1::Array{TV,0}
end

function HamiltonianVelocity3D(H::Hamiltonian{gauge,T}; vcomp=:whole) where {gauge,T}
    v32 = similar(H.h2)
    v31 = similar(H.h1)
    v21 = similar(H.h1)
    TH, TV = band_velocities_type(gauge, Val(vcomp), T)
    v30 = Array{TV,0}(undef)
    v20 = Array{TV,0}(undef)
    v10 = Array{TV,0}(undef)
    HamiltonianVelocity3D{Val{vcomp}(),gauge,T,TV,TH}(H, v32, v31, v21, v30, v20, v10)
end
#=
AutoBZ.period(b::HamiltonianVelocity3D) = period(b.H)
AutoBZ.coefficient_type(::Type{<:HamiltonianVelocity3D{kind,T}}) where {kind,T} = T
AutoBZ.fourier_type(::HamiltonianVelocity3D{kind,T,TV,TH}, _) where {kind,T,TV,TH} = Tuple{TH,TV,TV,TV}
AutoBZ.value(b::HamiltonianVelocity3D) = (value(b.H), only(b.vx_xyz), only(b.vy_xyz), only(b.vz_xyz))
@generated function AutoBZ.contract!(b::HamiltonianVelocity3D, x::Number, ::Type{Val{d}}) where d
    N = 3 # total dims
    quote
        # @nexprs doesn't like dot syntax in b.vi_j, so unpack the struct here
        Base.Cartesian.@nexprs $(N-d) $Symbol(:v, d-1) i -> b.$(Symbol(:v, d-1))_{N+1-i}
        # kernel calls
        ξ = inv(H.period[$d])
        fourier_kernel!($(Symbol(:(b.H.h), d-1)), $(Symbol(:(b.H.h), d)), x, ξ)
        Base.Cartesian.@nexprs $(N-d) i -> fourier_kernel!($(Symbol(:v, d-1))_{N+1-i}, $(Symbol(:v, d))_{N+1-i}, x, ξ)
        fourier_kernel!($(Symbol(:(b.v), d-1, :_, d)), $(Symbol(:(b.H.h), d)), x, ξ, Val{1}())
        return b
    end
end
AutoBZ.contract!(b::HamiltonianVelocity3D, x::Number, ::Type{Val{1}}) =
    (b.H.h0[], b.v30[], b.v20[], b.v10[] = b(x); return b)

function AutoBZ.contract!(b::HamiltonianVelocity3D{kind}, x::Number, ::) where kind
    if dim == 3
        ξ = inv(b.H.period[3])
        fourier_kernel!(b.H.h_z, b.H.h, x, ξ)
        fourier_kernel!(b.vz_z, b.H.h, x, ξ, Val{1}())
    elseif dim == 2
        ξ = inv(b.H.period[2])
        fourier_kernel!(b.H.h_yz, b.H.h_z, x, ξ)
        fourier_kernel!(b.vz_yz, b.vz_z, x, ξ)
        fourier_kernel!(b.vy_yz, b.H.h_z, x, ξ, Val{1}())
    elseif dim == 1
        b.H._xyz[], b.vz_xyz[], b.vy_xyz[], b.vx_xyz[] = b(x)
    else
        error("dim=$dim is out of bounds")
    end
    return b
end

function (b::HamiltonianVelocity3D{gauge,vcomp})(x::Number) where {gauge,vcomp}
    ξ = inv(b.H.period[1])
    band_velocities(gauge, vcomp,
        fourier_kernel(b.H.h1, x, ξ),
        fourier_kernel(b.v31,  x, ξ),
        fourier_kernel(b.v21,  x, ξ),
        fourier_kernel(b.H.h1, x, ξ, Val{1}()),
    )
end
=#
"""
    CovariantHamiltonianVelocity3D(H::HamiltonianVelocity3D{gauge}, Ax::F, Ay::F, Az::F) where {F<:FourierSeries3D}

Uses the Berry connection to return fully gauge-covariant velocities. Returns a
tuple of the Hamiltonian and the three velocity matrices.
"""
struct CovariantHamiltonianVelocity3D{vcomp,gauge,T,TV,TH,A} <: AbstractHamiltonianVelocity{vcomp,gauge,3,T}
    HV::HamiltonianVelocity3D{Val{:whole}(),Val{:Wannier},T,TV,TH}
    A_1::A
    A_2::A
    A_3::A
end
function CovariantHamiltonianVelocity3D(HV::V, A1::F, A2::F, A3::F; gauge=:Wannier, vcomp=:whole) where {T,TH,TV,V<:HamiltonianVelocity3D{Val{:whole}(),Val{:Wannier}(),T,TV,TH},F<:AbstractInplaceFourierSeries{3}}
    @assert period(H) == period(Ax) == period(Ay) == period(Az)
    CovariantHamiltonianVelocity3D{Val(vcomp),Val(gauge),T,TV,TH,F}(HV, A1, A2, A3)
end
#=
AutoBZ.period(b::CovariantHamiltonianVelocity3D) = period(b.H)
AutoBZ.coefficient_type(::Type{<:CovariantHamiltonianVelocity3D{kind,T}}) where {kind,T} = T
AutoBZ.fourier_type(::CovariantHamiltonianVelocity3D{kind,T,TA,TV,TH}, _) where {kind,T,TA,TV,TH} = Tuple{TH,TV,TV,TV}
AutoBZ.value(b::CovariantHamiltonianVelocity3D) = (value(b.H), only(b.vx_xyz), only(b.vy_xyz), only(b.vz_xyz))
function AutoBZ.contract!(b::CovariantHamiltonianVelocity3D, x::Number, dim)
    if dim == 3
        ξ = inv(b.H.period[3])
        fourier_kernel!(b.H.h_z, b.H.h, x, ξ)
        fourier_kernel!(b.Ax.coeffs_z, b.Ax.coeffs, x, ξ)
        fourier_kernel!(b.Ay.coeffs_z, b.Ay.coeffs, x, ξ)
        fourier_kernel!(b.Az.coeffs_z, b.Az.coeffs, x, ξ)
        fourier_kernel!(b.vz_z, b.H.h, x, ξ, Val{1}())
    elseif dim == 2
        ξ = inv(b.H.period[2])
        fourier_kernel!(b.H.h_yz, b.H.h_z, x, ξ)
        fourier_kernel!(b.Ax.coeffs_yz, b.Ax.coeffs_z, x, ξ)
        fourier_kernel!(b.Ay.coeffs_yz, b.Ay.coeffs_z, x, ξ)
        fourier_kernel!(b.Az.coeffs_yz, b.Az.coeffs_z, x, ξ)
        fourier_kernel!(b.vz_yz, b.vz_z, x, ξ)
        fourier_kernel!(b.vy_yz, b.H.h_z, x, ξ, Val{1}())
    elseif dim == 1
        b.H.h_xyz[], b.vz_xyz[], b.vy_xyz[], b.vx_xyz[] = b(x)
    else
        error("dim=$dim is out of bounds")
    end
    return b
end

function (b::CovariantHamiltonianVelocity3D{vcomp,gauge})(x::Number) where {vcomp,gauge}
    ξ = inv(b.H.period[1])
    H = fourier_kernel(b.H.h_yz, x, ξ)
    band_velocities(gauge, vcomp, H,
    # we take the Hermitian part of the Berry connection since Wannier 90 may have forgotten to do this
        velocity(H, fourier_kernel(b.vz_yz, x, ξ), herm(b.Az(x))),
        velocity(H, fourier_kernel(b.vy_yz, x, ξ), herm(b.Ay(x))),
        velocity(H, fourier_kernel(b.H.h_yz, x, ξ, Val{1}()), herm(b.Ax(x))),
    )
end
=#
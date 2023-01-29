export AbstractBandVelocity3D, BandEnergyVelocity3D, BandEnergyBerryVelocity3D

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
    band_velocities(::Type{Val{kind}}, H, vs...) where kind

Transform the band velocities according to the following values of `kind`
- `:orbital`: return the orbital-basis velocity (i.e. null-op, stays in Wannier gauge)
- `:band`: return the band-basis velocity (transforms to Hamiltonian gauge)
- `:intraband`: return only the diagonal of the band velocity (Hamiltonian gauge)
- `:interband`: return only the off-diagonal terms of the band velocity (Hamiltonian gauge)
"""
band_velocities(kind::T, H, vs::AbstractMatrix...) where T = band_velocities(kind, H, vs)

band_velocities(::Val{:orbital}, H, vs::NTuple) = (H, vs...)
band_velocities(::Val{:band}, H, vws::NTuple) = to_hamiltonian_gauge(H, vws)

function band_velocities(::Val{:interband}, H, vws::NTuple{N,T}) where {N,T}
    ϵ, vhs... = to_hamiltonian_gauge(H, vws)
    return (ϵ, ntuple(n -> vhs[n] - Diagonal(vhs[n]), Val{N}())...)
end

function band_velocities(::Val{:intraband}, H, vws::NTuple{N}) where N
    ϵ, vhs... = to_hamiltonian_gauge(H, vws)
    return (ϵ, ntuple(n -> Diagonal(vhs[n]), Val{N}())...)
end

function to_hamiltonian_gauge(H, vws::NTuple{N}) where {N}
    ishermitian(H) || throw(ArgumentError("found non-Hermitian Hamiltonian"))
    vals, U = eigen(Hermitian(H)) # need to wrap with Hermitian for type stability
    (Diagonal(vals), ntuple(n -> U'*vws[n]*U, Val{N}())...)
end

abstract type AbstractBandVelocity3D <:AbstractFourierSeries3D end

"""
    BandEnergyVelocity3D(coeffs, [period=(1.0,1.0,1.0), kind=:orbital])
    BandEnergyVelocity3D(H::FourierSeries3D, [kind=:orbital])

The in-place equivalent of `BandEnergyVelocity` for 3D series evaluation.
"""
struct BandEnergyVelocity3D{kind,T,TV,TH} <: AbstractBandVelocity3D
    H::FourierSeries3D{T,0,0,0}
    vz_z::Array{T,2}
    vz_yz::Array{T,1}
    vy_yz::Array{T,1}
    vz_xyz::Array{TV,0}
    vy_xyz::Array{TV,0}
    vx_xyz::Array{TV,0}
    H_xyz::Array{TH,0}
end

BandEnergyVelocity3D(coeffs, period=(1.0,1.0,1.0), kind=:orbital) = BandEnergyVelocity3D(FourierSeries3D(coeffs, period), kind)
function BandEnergyVelocity3D(H::FourierSeries3D{T,0,0,0}, kind=:orbital) where T
    vz_z   = similar(H.coeffs_z)
    vz_yz  = similar(H.coeffs_yz)
    vy_yz  = similar(H.coeffs_yz)
    TH, TV = band_velocity_types(kind, T)
    vz_xyz = Array{TV,0}(undef)
    vy_xyz = Array{TV,0}(undef)
    vx_xyz = Array{TV,0}(undef)
    H_xyz  = Array{TH,0}(undef)
    BandEnergyVelocity3D{Val{kind}(),T,TV,TH}(H, vz_z, vz_yz, vy_yz, vz_xyz, vy_xyz, vx_xyz, H_xyz)
end

function band_velocity_types(kind, T)
    if kind == :orbital
        return T, T
    elseif kind == :band || kind == :interband
        return diagonal_type(real(T)), T
    elseif kind == :intraband
        return diagonal_type(real(T)), diagonal_type(T)
    else
        error("band velocity kind not recognized")
    end
end

AutoBZ.period(b::BandEnergyVelocity3D) = period(b.H)
AutoBZ.coefficient_type(::Type{<:BandEnergyVelocity3D{kind,T}}) where {kind,T} = T
AutoBZ.fourier_type(::BandEnergyVelocity3D{kind,T,TV,TH}, _) where {kind,T,TV,TH} = Tuple{TH,TV,TV,TV}
AutoBZ.value(b::BandEnergyVelocity3D) = (only(b.H_xyz), only(b.vx_xyz), only(b.vy_xyz), only(b.vz_xyz))
function contract!(b::BandEnergyVelocity3D{kind}, x::Number, dim) where kind
    if dim == 3
        ξ = inv(b.H.period[3])
        fourier_kernel!(b.H.coeffs_z, b.H.coeffs, x, ξ)
        fourier_kernel!(b.vz_z, b.H.coeffs, x, ξ, Val{1}())
    elseif dim == 2
        ξ = inv(b.H.period[2])
        fourier_kernel!(b.H.coeffs_yz, b.H.coeffs_z, x, ξ)
        fourier_kernel!(b.vz_yz, b.vz_z, x, ξ)
        fourier_kernel!(b.vy_yz, b.H.coeffs_z, x, ξ, Val{1}())
    elseif dim == 1
        ξ = inv(b.H.period[1])
        b.H_xyz[], b.vz_xyz[], b.vy_xyz[], b.vx_xyz[] = 
            band_velocities(kind,
                fourier_kernel(b.H.coeffs_yz, x, ξ),
                fourier_kernel(b.vz_yz, x, ξ),
                fourier_kernel(b.vy_yz, x, ξ),
                fourier_kernel(b.H.coeffs_yz, x, ξ, Val{1}()),
            )
    else
        error("dim=$dim is out of bounds")
    end
    return b
end

"""
    BandEnergyBerryVelocity3D(H::FourierSeries3D{TH,0,0,0}, Ax::FourierSeries3D{TA,0,0,0}, Ay::FourierSeries3D{TA,0,0,0}, Az::FourierSeries3D{TA,0,0,0}, [kind=:orbital]) where {TH,TA}

The in-place equivalent of `BandEnergyBerryVelocity` for 3D series evaluation.
"""
struct BandEnergyBerryVelocity3D{kind,T,TA,TV,TH} <: AbstractBandVelocity3D
    H::FourierSeries3D{T,0,0,0}
    Ax::FourierSeries3D{TA,0,0,0}
    Ay::FourierSeries3D{TA,0,0,0}
    Az::FourierSeries3D{TA,0,0,0}
    vz_z::Array{T,2}
    vz_yz::Array{T,1}
    vy_yz::Array{T,1}
    vz_xyz::Array{TV,0}
    vy_xyz::Array{TV,0}
    vx_xyz::Array{TV,0}
    H_xyz::Array{TH,0}
end
function BandEnergyBerryVelocity3D(H::FourierSeries3D{T,0,0,0}, Ax::FourierSeries3D{TA,0,0,0}, Ay::FourierSeries3D{TA,0,0,0}, Az::FourierSeries3D{TA,0,0,0}, kind=:orbital) where {T,TA}
    @assert period(H) == period(Ax) == period(Ay) == period(Az)
    vz_z   = similar(H.coeffs_z)
    vz_yz  = similar(H.coeffs_yz)
    vy_yz  = similar(H.coeffs_yz)
    TH, TV = band_velocity_types(kind, promote_type(T, TA))
    vz_xyz = Array{TV,0}(undef)
    vy_xyz = Array{TV,0}(undef)
    vx_xyz = Array{TV,0}(undef)
    H_xyz  = Array{TH,0}(undef)
    BandEnergyBerryVelocity3D{Val{kind}(),T,TA,TV,TH}(H, Ax, Ay, Az, vz_z, vz_yz, vy_yz, vz_xyz, vy_xyz, vx_xyz, H_xyz)
end

AutoBZ.period(b::BandEnergyBerryVelocity3D) = period(b.H)
AutoBZ.coefficient_type(::Type{<:BandEnergyBerryVelocity3D{kind,T}}) where {kind,T} = T
AutoBZ.fourier_type(::BandEnergyBerryVelocity3D{kind,T,TA,TV,TH}, _) where {kind,T,TA,TV,TH} = Tuple{TH,TV,TV,TV}
AutoBZ.value(b::BandEnergyBerryVelocity3D) = (only(b.H_xyz), only(b.vx_xyz), only(b.vy_xyz), only(b.vz_xyz))
function contract!(b::BandEnergyBerryVelocity3D{kind}, x::Number, dim) where kind
    if dim == 3
        ξ = inv(b.H.period[3])
        fourier_kernel!(b.H.coeffs_z, b.H.coeffs, x, ξ)
        fourier_kernel!(b.Ax.coeffs_z, b.Ax.coeffs, x, ξ)
        fourier_kernel!(b.Ay.coeffs_z, b.Ay.coeffs, x, ξ)
        fourier_kernel!(b.Az.coeffs_z, b.Az.coeffs, x, ξ)
        fourier_kernel!(b.vz_z, b.H.coeffs, x, ξ, Val{1}())
    elseif dim == 2
        ξ = inv(b.H.period[2])
        fourier_kernel!(b.H.coeffs_yz, b.H.coeffs_z, x, ξ)
        fourier_kernel!(b.Ax.coeffs_yz, b.Ax.coeffs_z, x, ξ)
        fourier_kernel!(b.Ay.coeffs_yz, b.Ay.coeffs_z, x, ξ)
        fourier_kernel!(b.Az.coeffs_yz, b.Az.coeffs_z, x, ξ)
        fourier_kernel!(b.vz_yz, b.vz_z, x, ξ)
        fourier_kernel!(b.vy_yz, b.H.coeffs_z, x, ξ, Val{1}())
    elseif dim == 1
        ξ = inv(b.H.period[1])
        H = fourier_kernel(b.H.coeffs_yz, x, ξ)
        b.H_xyz[], b.vz_xyz[], b.vy_xyz[], b.vx_xyz[] = 
        band_velocities(kind, H,
        # we take the Hermitian part of the Berry connection since Wannier 90 may have forgotten to do this
            velocity(H, fourier_kernel(b.vz_yz, x, ξ), herm(b.Az(x))),
            velocity(H, fourier_kernel(b.vy_yz, x, ξ), herm(b.Ay(x))),
            velocity(H, fourier_kernel(b.H.coeffs_yz, x, ξ, Val{1}()), herm(b.Ax(x))),
        )
    else
        error("dim=$dim is out of bounds")
    end
    return b
end

function shift!(HV::Union{BandEnergyVelocity3D,BandEnergyBerryVelocity3D}, λ)
    shift!(HV.H, λ)
    return HV
end
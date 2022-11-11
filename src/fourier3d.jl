#=
This file contains specialized in-place Fourier series evaluators that define
the method contract!, instead of contract, and are limited to 3D series.
=#

export AbstractFourierSeries3D, contract!,
    FourierSeries3D, BandEnergyVelocity3D, BandEnergyBerryVelocity3D

"""
    AbstractFourierSeries3D <: AbstractFourierSeries{3}

An abstract subtype of `AbstractFourierSeries{3}` representing in-place Fourier
series evaluators
"""
abstract type AbstractFourierSeries3D <: AbstractFourierSeries{3} end

"""
    contract!(f::AbstractFourierSeries3D, x::Number, dim::Int)


"""
function contract! end

contract(f::AbstractFourierSeries3D, x, dim) = contract!(f, x, dim)

(f::AbstractFourierSeries3D)(x::SVector{N}) where N = (contract!(f, x[N], N); f(pop(x)))
(f::AbstractFourierSeries3D)(x::SVector{1}) = f(only(x))
(f::AbstractFourierSeries3D)(x::Number) = value(contract!(f, x, 1))

"""
    fourier_kernel!(r::Array{T,N-1}, C::Array{T,N}, x, ξ, [::Val{a}=Val{0}()]) where {T,N,a}

Contract the outermost index of array `C` and write it to the array `r`. Assumes
the size of the outermost dimension of `C` is `2m+1` and sums the coefficients
```math
r_{i_{1},\\dots,i_{N-1}} = \\sum_{i_{N}=-m}^{m} C_{i_{1},\\dots,i_{N-1},i_{N}+m+1} (i2\\pi\\xi i_{N})^{a} \\exp(i2\\pi\\xi x i_{N})
```
Hence this represents evaluation of a Fourier series with `m` modes. The
parameter `a` represents the order of derivative of the Fourier series.
"""
@generated function fourier_kernel!(r::Array{T,N_}, C::Array{T,N}, x, ξ, ::Val{a}=Val{0}()) where {T,N,N_,a}
    N != N_+1 && return :(error("array dimensions incompatible"))
    if a == 0
        fundamental = :(Base.Cartesian.@nref $N C d -> d == $N ? m+1 : i_d)
        c = :(z); c̄ = :(conj(z))
    elseif a == 1
        fundamental = :(zero($T))
        c = :(im*2π*ξ*n*z); c̄ = :(conj(c))
    else
        f₀ = 0^a
        fundamental = :(fill($f₀, $T))
        c = :(((im*2pi*ξ*n)^$a)*z); c̄ = :(((-im*2pi*ξ*n)^$a)*conj(z))
    end
quote
    size(r) == size(C)[1:$N_] || error("array sizes incompatible")
    s = size(C,$N)
    isodd(s) || return error("expected an array with an odd number of coefficients")
    m = div(s,2)
    @inbounds Base.Cartesian.@nloops $N_ i r begin
        (Base.Cartesian.@nref $N_ r i) = $fundamental
    end
    z₀ = cispi(2ξ*x)
    z = one(z₀)
    for n in Base.OneTo(m)
        z *= z₀
        c  = $c
        c̄  = $c̄
        @inbounds Base.Cartesian.@nloops $N_ i r begin
            (Base.Cartesian.@nref $N_ r i) += c*(Base.Cartesian.@nref $N C d -> d == $N ? n+m+1 : i_d) + c̄*(Base.Cartesian.@nref $N C d -> d == $N ? -n+m+1 : i_d)
        end
    end
end
end


"""
    FourierSeries3D(coeffs::Array{T,3}, [period=ones(SVector{3,Float64})])

This type is an `AbstractFourierSeries{3}` designed for in-place evaluation of
`FourierSeries`, and unlike `FourierSeries` is specialized for 3D Fourier series
and does not allocate a new array every time `contract` is called on it. This
type stores the intermediate arrays used in a calculation and assumes that the
size of `coeffs` on each axis is odd because it treats the zeroth harmonic as
the center of the array (i.e. `(size(coeffs) .÷ 2) .+ 1`).
"""
struct FourierSeries3D{T,a1,a2,a3} <: AbstractFourierSeries3D
    coeffs::Array{T,3}
    period::NTuple{3,Float64}
    coeffs_z::Array{T,2}
    coeffs_yz::Array{T,1}
    coeffs_xyz::Array{T,0}
end

function FourierSeries3D(coeffs::Array{T,3}, period=(1.0,1.0,1.0), orders=(0,0,0)) where T
    @assert all(map(isodd, size(coeffs)))
    coeffs_z = Array{T,2}(undef, size(coeffs,1), size(coeffs, 2))
    coeffs_yz = Array{T,1}(undef, size(coeffs,1))
    coeffs_xyz = Array{T,0}(undef)
    FourierSeries3D{T,orders...}(coeffs, period, coeffs_z, coeffs_yz, coeffs_xyz)
end
FourierSeries3D(f::FourierSeries{3}) = FourierSeries3D(collect(f.coeffs), Tuple(f.period))

period(f::FourierSeries3D) = f.period
Base.eltype(::Type{<:FourierSeries3D{T}}) where T = T
value(f::FourierSeries3D) = only(f.coeffs_xyz)
function contract!(f::FourierSeries3D{T,a1,a2,a3}, x::Number, dim) where {T,a1,a2,a3}
    if dim == 3
        fourier_kernel!(f.coeffs_z, f.coeffs, x, inv(f.period[3]), Val{a3}())
    elseif dim == 2
        fourier_kernel!(f.coeffs_yz, f.coeffs_z, x, inv(f.period[2]), Val{a2}())
    elseif dim == 1
        fourier_kernel!(f.coeffs_xyz, f.coeffs_yz, x, inv(f.period[1]), Val{a1}())
    else
        error("dim=$dim is out of bounds")
    end
    return f
end

function (f::FourierSeries3D{T,0})(x::Number) where T
    C = f.coeffs_yz
    ξ = inv(f.period[1])
    s = size(C,1)
    isodd(s) || return error("expected an array with an odd number of coefficients")
    m = div(s,2)
    @inbounds r = C[m+1]
    z₀ = cispi(2ξ*x)
    z = one(z₀)
    @inbounds for n in Base.OneTo(m)
        z *= z₀
        r += z*C[n+m+1] + conj(z)*C[-n+m+1]
    end
    r
end

"""
"""
struct BandEnergyVelocity3D{T} <: AbstractFourierSeries3D
    H::FourierSeries3D{T,0,0,0}
    vz_z::Array{T,2}
    vz_yz::Array{T,1}
    vy_yz::Array{T,1}
    vz_xyz::Array{T,0}
    vy_xyz::Array{T,0}
    vx_xyz::Array{T,0}
end

BandEnergyVelocity3D(coeffs, period=(1.0,1.0,1.0)) = BandEnergyVelocity3D(FourierSeries3D(coeffs, period))
BandEnergyVelocity3D(coeffs::FourierSeries{3}) = BandEnergyVelocity3D(FourierSeries3D(coeffs))
function BandEnergyVelocity3D(H::FourierSeries3D{T,0,0,0}) where T
    vz_z = similar(H.coeffs_z)
    vz_yz = similar(H.coeffs_yz)
    vy_yz = similar(H.coeffs_yz)
    vz_xyz = similar(H.coeffs_xyz)
    vy_xyz = similar(H.coeffs_xyz)
    vx_xyz = similar(H.coeffs_xyz)
    BandEnergyVelocity3D{T}(H, vz_z, vz_yz, vy_yz, vz_xyz, vy_xyz, vx_xyz)
end

period(b::BandEnergyVelocity3D) = period(b.H)
Base.eltype(::Type{BandEnergyVelocity3D{T}}) where T = NTuple{4,T}
value(b::BandEnergyVelocity3D) = (value(b.H), only(b.vx_xyz), only(b.vy_xyz), only(b.vz_xyz))
function contract!(b::BandEnergyVelocity3D{T}, x::Number, dim) where T
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
        fourier_kernel!(b.H.coeffs_xyz, b.H.coeffs_yz, x, ξ)
        fourier_kernel!(b.vz_xyz, b.vz_yz, x, ξ)
        fourier_kernel!(b.vy_xyz, b.vy_yz, x, ξ)
        fourier_kernel!(b.vx_xyz, b.H.coeffs_yz, x, ξ, Val{1}())
    else
        error("dim=$dim is out of bounds")
    end
    return b
end

"""
"""
struct BandEnergyBerryVelocity3D{T,TA} <: AbstractFourierSeries3D
    H::FourierSeries3D{T,0,0,0}
    Ax::FourierSeries3D{TA,0,0,0}
    Ay::FourierSeries3D{TA,0,0,0}
    Az::FourierSeries3D{TA,0,0,0}
    vz_z::Array{T,2}
    vz_yz::Array{T,1}
    vy_yz::Array{T,1}
    vz_xyz::Array{T,0}
    vy_xyz::Array{T,0}
    vx_xyz::Array{T,0}
end

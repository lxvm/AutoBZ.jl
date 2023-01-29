#=
This file contains specialized in-place Fourier series evaluators that define
the method contract!, instead of contract, and are limited to 3D series.
=#

export AbstractFourierSeries3D, contract!, shift!, FourierSeries3D

"""
    AbstractFourierSeries3D <: AbstractFourierSeries{3}

An abstract subtype of `AbstractFourierSeries{3}` representing in-place Fourier
series evaluators
"""
abstract type AbstractFourierSeries3D <: AbstractFourierSeries{3} end

"""
    contract!(f::AbstractFourierSeries3D, x::Number, dim::Int)

An in-place version of `contract`.
"""
function contract! end

AutoBZ.contract(f::AbstractFourierSeries3D, x, dim) = contract!(f, x, dim)

(f::AbstractFourierSeries3D)(x::SVector{N}) where N = (contract!(f, x[N], N); f(pop(x)))
(f::AbstractFourierSeries3D)(x::SVector{1}) = f(only(x))
(f::AbstractFourierSeries3D)(x::Number) = value(contract!(f, x, 1))

AutoBZ.iterated_pre_eval(f::Union{FourierIntegrand{F,S},IteratedFourierIntegrand{F,S}}, x, dim::Int) where {F,S<:AbstractFourierSeries3D} =
    (contract!(f.s, x, dim); return f)
#=
function AutoBZ.fourier_pre_eval(f::AbstractFourierSeries3D, bz::FullBZ{<:Any,3}, npt)
    # @assert collect(period(f)) ≈ [x[2] - x[1] for x in boundingbox(bz)] "Integration region doesn't match integrand period"
    pre = Array{Tuple{fourier_type(f, domain_type(bz)),Int},3}(undef, npt, npt, npt)
    box = boundingbox(bz)
    for k in 1:npt
        contract!(f, (box[3][2]-box[3][1])*(k-1)/npt + box[3][1], 3)
            for j in 1:npt
            contract!(f, (box[2][2]-box[2][1])*(j-1)/npt + box[2][1], 2)
            for i in 1:npt
                pre[i,j,k] = (f((box[1][2]-box[1][1])*(i-1)/npt + box[1][1]), 1)
            end
        end
    end
    return pre
end

function AutoBZ.fourier_pre_eval(f::AbstractFourierSeries3D, bz::AbstractBZ{3}, npt)
    # @assert collect(period(f)) ≈ [x[2] - x[1] for x in boundingbox(bz)] "Integration region doesn't match integrand period"
    flag, wsym, nsym = equispace_rule(bz, npt)
    n = 0
    box = boundingbox(bz)
    pre = Vector{Tuple{fourier_type(f, domain_type(bz)),Int}}(undef, nsym)
    for k in axes(flag, 3)
        contract!(f, (box[3][2]-box[3][1])*(k-1)/npt, 3)
        for j in axes(flag, 2)
            contract!(f, (box[2][2]-box[2][1])*(j-1)/npt, 2)
            for i in axes(flag, 1)
                if flag[i,j,k]
                    n += 1
                    pre[n] = (f((box[1][2]-box[1][1])*(i-1)/npt), wsym[n])
                    n >= nsym && break
                end
            end
        end
    end
    return pre
end
=#
"""
    FourierSeries3D(coeffs::Array{T,3}, [period=(1.0, 1.0, 1.0)])

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

AutoBZ.period(f::FourierSeries3D) = f.period
AutoBZ.coefficient_type(::Type{<:FourierSeries3D{T}}) where T = T
AutoBZ.value(f::FourierSeries3D) = only(f.coeffs_xyz)
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

(f::FourierSeries3D{T,a1})(x::Number) where {T,a1} = fourier_kernel(f.coeffs_yz, x, inv(f.period[1]), Val{a1}())

"""
    shift!(f::FourierSeries3D, λ::Number)

Modifies and returns `f` such that it returns `f - λ*I`.
"""
function shift!(f::FourierSeries3D{T}, λ_::Number) where T
    λ = convert(eltype(T), λ_)
    i = div.(size(f.coeffs), 2) .+ 1
    f.coeffs[i...] -= λ*I
    return f
end

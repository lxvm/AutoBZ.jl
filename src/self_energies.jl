export AbstractSelfEnergy, lb, ub
export EtaSelfEnergy, ConstScalarSelfEnergy, ScalarSelfEnergy, MatrixSelfEnergy

"""
    AbstractSelfEnergy

An abstract type whose instances implement the following interface:
- instances are callable and return a square matrix as a function of frequency
- instances have methods `lb` and `ub` that return the lower and upper bounds of
  of the frequency domain for which the instance can be evaluated
"""
abstract type AbstractSelfEnergy end

"""
    lb(::AbstractSelfEnergy)

Return the greatest lower bound of the domain of the self energy evaluator
"""
function lb end


"""
    ub(::AbstractSelfEnergy)

Return the least upper bound of the domain of the self energy evaluator
"""
function ub end

"""
    ConstScalarSelfEnergy(v::Number)

Construct a self-energy evaluator which returns ``v I`` for any frequency.
"""
struct ConstScalarSelfEnergy{T<:Number} <: AbstractSelfEnergy
    v::T
end
(Σ::ConstScalarSelfEnergy)(::Number) = Σ.v*I
lb(::ConstScalarSelfEnergy) = -Inf
ub(::ConstScalarSelfEnergy) = Inf

"""
    EtaSelfEnergy(η::Real)

Construct a [`ConstScalarSelfEnergy`](@ref) with value `-im*η`.
"""
EtaSelfEnergy(η::Real) = ConstScalarSelfEnergy(-im*η)

"""
    ScalarSelfEnergy(interpolant, lb, ub)

Construct a self-energy evaluator which for frequencies above `lb` and below
`ub` returns the interpolant at that frequency times an identity matrix.
"""
struct ScalarSelfEnergy{T} <: AbstractSelfEnergy
    interpolant::T
    lb::Float64
    ub::Float64
end
(Σ::ScalarSelfEnergy)(ω::Number) = Σ.interpolant(ω)*I
lb(Σ::ScalarSelfEnergy) = Σ.lb
ub(Σ::ScalarSelfEnergy) = Σ.ub


"""
    MatrixSelfEnergy(interpolant, lb, ub)

Construct a self-energy evaluator which for frequencies above `lb` and below
`ub` returns the matrix-valued interpolant at that frequency.
"""
struct MatrixSelfEnergy{T} <: AbstractSelfEnergy
    interpolant::T
    lb::Float64
    ub::Float64
end
(Σ::MatrixSelfEnergy)(ω::Number) = Σ.interpolant(ω)
lb(Σ::MatrixSelfEnergy) = Σ.lb
ub(Σ::MatrixSelfEnergy) = Σ.ub



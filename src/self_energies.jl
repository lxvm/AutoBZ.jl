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
`ub` returns the scalar interpolant at that frequency times an identity matrix.
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
    DiagonalSelfEnergy(interpolant, lb, ub)

Construct a self-energy evaluator which for frequencies above `lb` and below
`ub` returns the vector interpolant at that frequency wrapped by a `Diagonal`.
"""
struct DiagonalSelfEnergy{T} <: AbstractSelfEnergy
    interpolant::T
    lb::Float64
    ub::Float64
end
(Σ::DiagonalSelfEnergy)(ω::Number) = Diagonal(Σ.interpolant(ω))
lb(Σ::DiagonalSelfEnergy) = Σ.lb
ub(Σ::DiagonalSelfEnergy) = Σ.ub


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



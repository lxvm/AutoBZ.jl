"""
    ConstScalarSelfEnergy(v::Number)

Construct a self-energy evaluator which returns ``v I`` for any frequency.
"""
struct ConstScalarSelfEnergy{T<:Number,F} <: AbstractSelfEnergy
    v::T
    lb::F
    ub::F
end
function ConstScalarSelfEnergy(v)
    T = typeof(real(v))
    return ConstScalarSelfEnergy(v, -typemax(T), typemax(T))
end
(Σ::ConstScalarSelfEnergy)(::Number) = Σ.v*I
lb(Σ::ConstScalarSelfEnergy) = Σ.lb
ub(Σ::ConstScalarSelfEnergy) = Σ.ub

"""
    EtaSelfEnergy(η::Number)

Construct a [`ConstScalarSelfEnergy`](@ref) with value `-im*η`.
"""
EtaSelfEnergy(η::Number) = ConstScalarSelfEnergy(-im*η)

"""
    ScalarSelfEnergy(interpolant, lb, ub)

Construct a self-energy evaluator which for frequencies above `lb` and below
`ub` returns the scalar interpolant at that frequency times an identity matrix.
"""
struct ScalarSelfEnergy{T,F} <: AbstractSelfEnergy
    interpolant::T
    lb::F
    ub::F
end
(Σ::ScalarSelfEnergy)(ω::Number) = Σ.interpolant(ω)*I
lb(Σ::ScalarSelfEnergy) = Σ.lb
ub(Σ::ScalarSelfEnergy) = Σ.ub


"""
    DiagonalSelfEnergy(interpolant, lb, ub)

Construct a self-energy evaluator which for frequencies above `lb` and below
`ub` returns the vector interpolant at that frequency wrapped by a `Diagonal`.
"""
struct DiagonalSelfEnergy{T,F} <: AbstractSelfEnergy
    interpolant::T
    lb::F
    ub::F
end
(Σ::DiagonalSelfEnergy)(ω::Number) = Diagonal(Σ.interpolant(ω))
lb(Σ::DiagonalSelfEnergy) = Σ.lb
ub(Σ::DiagonalSelfEnergy) = Σ.ub


"""
    MatrixSelfEnergy(interpolant, lb, ub)

Construct a self-energy evaluator which for frequencies above `lb` and below
`ub` returns the matrix-valued interpolant at that frequency.
"""
struct MatrixSelfEnergy{T,F} <: AbstractSelfEnergy
    interpolant::T
    lb::F
    ub::F
end
(Σ::MatrixSelfEnergy)(ω::Number) = Σ.interpolant(ω)
lb(Σ::MatrixSelfEnergy) = Σ.lb
ub(Σ::MatrixSelfEnergy) = Σ.ub

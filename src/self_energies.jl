export AbstractSelfEnergy, lb, ub, EtaEnergy, ScalarEnergy

"""
    AbstractSelfEnergy

An abstract type whose instances implement the following interface:
- instances are callable and return a square matrix as a function of frequency
- instances have methods `lb` and `ub` that return the lower and upper bounds of
  of the frequency domain for which the instance can be evaluated
"""
abstract type AbstractSelfEnergy end
function lb end
function ub end

"""
    EtaEnergy(η::Real)

Construct a self-energy evaluator which returns ``-i\\eta I`` for any frequency.
"""
struct EtaEnergy{T<:Real} <: AbstractSelfEnergy
    η::T
end
(Σ::EtaEnergy)(::Number) = complex(zero(Σ.η), -Σ.η)*I
lb(::EtaEnergy) = -Inf
ub(::EtaEnergy) = Inf

"""
    ScalarEnergy(interpolant, lb, ub)

Construct a self-energy evaluator which for frequencies above `lb` and below
`ub` returns the interpolant at that frequency times an identity matrix.
"""
struct ScalarEnergy{T} <: AbstractSelfEnergy
    interpolant::T
    lb::Float64
    ub::Float64
end
(Σ::ScalarEnergy)(ω::Number) = Σ.interpolant(ω)*I
lb(Σ::ScalarEnergy) = Σ.lb
ub(Σ::ScalarEnergy) = Σ.ub
"""
    AbstractSelfEnergy

Interface: instances are callable and return a matrix as a function of frequency.
"""
abstract type AbstractSelfEnergy end

struct EtaEnergy{T<:Real} <: AbstractSelfEnergy
    η::T
end
(Σ::EtaEnergy)(::Number) = complex(zero(Σ.η), -Σ.η)*I

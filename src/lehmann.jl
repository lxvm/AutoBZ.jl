abstract type AbstractDLRAlgorithm end

struct DLRIntegral{T} <: AbstractDLRAlgorithm
    Λ::T
end

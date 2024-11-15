abstract type AbstractDLRAlgorithm end

"""
    LehmannJL(; Λ, rtol)

Do `using Lehmann` to use this algorithm for the electron density, which uses the DLR and
evaluates the Green's function on the imaginary axis
"""
struct LehmannJL{T,R,K} <: AbstractDLRAlgorithm
    Λ::T
    rtol::R
    isFermi::Bool
    kws::K
end
LehmannJL(; Λ, rtol=1e-8, isFermi=true, kws...) = LehmannJL(Λ, rtol, isFermi, kws)

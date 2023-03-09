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
    AbstractWannierInterp{G,N} <: AbstractInplaceFourierSeries{N}

An abstract subtype of `AbstractInplaceFourierSeries` representing in-place
Fourier series evaluators for Wannier-interpolated quantities with a choice of
basis, or gauge, `G`, which is typically `Val(:Hamiltonian)` or `Val(:Wannier)`.
For details, see [`to_gauge`](@ref).
"""
abstract type AbstractWannierInterp{G,N,T} <: AbstractFourierSeries{N,T} end

gauge(::AbstractWannierInterp{G}) where G = G


"""
    hamiltonian(::AbstractWannierInterp)

Return the Hamiltonian object used for Wannier interpolation
"""
function hamiltonian end

coefficients(w::AbstractWannierInterp) = coefficients(hamiltonian(w))

"""
    shift!(::AbstractWannierInterp, λ)

Offset the zero-point energy in a Hamiltonian system by a constant
"""
function shift!(w::AbstractWannierInterp, λ)
    shift!(hamiltonian(w), λ)
    return w
end

period(w::AbstractWannierInterp) = period(hamiltonian(w))

show_details(w::AbstractWannierInterp) =
    " & $(gauge(w)) gauge"

"""
    AbstractVelocity{C,G,N,T} <:AbstractWannierInterp{G,N,T}

An abstract substype of `AbstractWannierInterp` also containing information
about the velocity component, `C`, which is typically `Val(:whole)`,
`Val(:inter)`, or `Val(:intra)`. For details see [`to_vcomp_gauge`](@ref).
"""
abstract type AbstractVelocity{C,G,N,T} <:AbstractWannierInterp{G,N,T} end

vcomp(::AbstractVelocity{C}) where C = C

show_details(v::AbstractVelocity) =
    " & $(vcomp(v)) velocity component  & $(gauge(v)) gauge"
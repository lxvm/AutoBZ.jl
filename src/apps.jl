# The integrands here all define a set of canonical parameters
# Typically we also want to transform the problem based on the parameters, e.g.
# - precompute some function that only depends on parameters, not variables
# - truncate the limits of frequency integration
# - set a Δn for PTR
# The infrastructure for parameter transformations provides a service to the user for
# convenience (they don't have to do these transformations themselves) and performance (it
# lets us pre-allocate parallelization buffers and rule caches) with the constraint that the
# types of the parameters the user provides is limited to a canonical set (so that return
# types of the integrand are predictable). We use these facilities to provide a high-level
# interface for pre-defined problems to the generic and extensible AutoBZCore library.

# IntegralSolver defines two codepaths since it is oblivious to parameters:
# 1. make_cache for integrands that don't define `init_solver_cacheval` (and variants)
# 2. remake_cache for integrands that do
# We rely on path 2 for parameter-based transformations since it is the only way to
# repeat the transformation for all parameters (including the first). Since the parameters
# haven't been given when the IntegralSolver is made, we redirect `init_solver_cacheval` to
# `init_cacheval` with the following parameter type so we can dispatch at
# `integrand_return_type` to

struct CanonicalParameters end

#= these need to be defined for each of the integrands
function AutoBZCore.init_solver_cacheval(f, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end
function AutoBZCore.init_fourier_solver_cacheval(s::AbstractFourierSeries, f, dom, alg)
    return AutoBZCore.init_fourier_cacheval(s, f, dom, CanonicalParameters(), alg)
end
function AutoBZCore.integrand_return_type(f, x, ::CanonicalParameters)

end
=#
propagator_denominator(h, M) = M-h
propagator_denominator(h::Eigen, M::UniformScaling) =
    propagator_denominator(Diagonal(h.values), M)
propagator_denominator((h, U)::Eigen, M::AbstractMatrix) =
    propagator_denominator(Diagonal(h), U' * M * U) # rotate to Hamiltonian gauge
propagator_denominator(h::FourierValue, M) = propagator_denominator(h.s, M)

"""
    gloc_integrand(h, M)

Returns `inv(M-h)` where `M = ω*I-Σ(ω)`
"""
gloc_integrand(G) = inv(G)

"""
    diaggloc_integrand(h, M)

Returns `diag(inv(M-h))` where `M = ω*I-Σ(ω)`
"""
diaggloc_integrand(G) = diag_inv(G)

"""
    trgloc_integrand(h, M)

Returns `tr(inv(M-h))` where `M = ω*I-Σ(ω)`
"""
trgloc_integrand(G) = tr_inv(G)

"""
    dos_integrand(h, M)

Returns `imag(tr(inv(M-h)))/(-pi)` where `M = ω*I-Σ(ω)`. It is unsafe to use
this in the inner integral for small eta due to the localized tails of the
integrand. The default, safe version also integrates the real part which is less
localized, at the expense of a slight slow-down due to complex arithmetic.
See [`AutoBZ.Jobs.safedos_integrand`](@ref).
"""
dos_integrand(G) = imag(tr_inv(G))/(-pi)

# shift energy by chemical potential, but not self energy
# evalM(M::Union{AbstractMatrix,UniformScaling}) = (M,) # don't provide this method since not canonical
evalM(Σ::Union{AbstractMatrix,UniformScaling}, ω, μ) = ((ω+μ)*I - Σ,)
evalM(Σ::AbstractSelfEnergy, ω, μ) = evalM(Σ(ω), ω, μ)
evalM(Σ, ω; μ=0) = evalM(Σ, ω, μ)
evalM(Σ; ω, μ=0) = evalM(Σ, ω, μ)

# given mixed parameters and a function that maps them to a tuple of a canonical
# form for the integrand, return new mixed parameters
function canonize(f, p::MixedParameters)
    params = f(getfield(p, :args)...; getfield(p, :kwargs)...)
    return MixedParameters(params, NamedTuple())
end

# Generic behavior for single Green's function integrands (methods and types)
for name in ("Gloc", "DiagGloc", "TrGloc", "DOS")
    # create and export symbols
    f = Symbol(lowercase(name), "_integrand")
    T = Symbol(name, "Integrand")

    # define a method to evaluate the Green's function
    @eval $f(h, M) = $f(propagator_denominator(h, M))

    # Define the type alias to have the same behavior as the function
    """
    $(T)(h, Σ, ω, μ)
    $(T)(h, Σ, ω; μ=0)
    $(T)(h, Σ; ω, μ=0)

    See `Integrand` for more details.
    """
    @eval function $T(h::HamiltonianInterp, Σ, args...; kwargs...)
        return FourierIntegrand($f, h, Σ, args...; kwargs...)
    end

    # We provide the following functions to build a cache during construction of IntegralSolvers
    @eval function AutoBZCore.init_fourier_solver_cacheval(s::AbstractFourierSeries, f::Integrand{typeof($f)}, dom, alg)
        new_alg = set_autoptr_eta(alg, f, f.p)
        return AutoBZCore.init_fourier_cacheval(s, f, dom, CanonicalParameters(), new_alg)
    end

    # evaluate the integrand once for the expected return type
    @eval function AutoBZCore.integrand_return_type(f::Integrand{typeof($f)}, x, ::CanonicalParameters)
        return typeof($f(x.s, evalM(f.p[1], 0.0)...))
    end

    @eval function AutoBZCore.remake_integrand_cache(f::Integrand{typeof($f)}, dom, p, alg, cacheval, kwargs)
        # pre-evaluate the self energy when remaking the cache
        new_p = canonize(evalM, p)
        # Define default equispace grid stepping
        new_alg, new_cacheval = reset_autoptr_eta(alg, cacheval, f, dom, new_p)
        return AutoBZCore.IntegralCache(f, dom, new_p, new_alg, new_cacheval, kwargs)
    end

    # estimate a value of eta that should suffice for most parameters
    # ideally we would have some upper bound on the gradient of the Hamiltonian, v,
    # divide by the largest period, T, and take a = η*v/T
    @eval function set_autoptr_eta(alg::AutoPTR, f::Integrand{typeof($f)}, p)
        # (estimated) eta from self energy evaluated at the Fermi energy
        η = im_sigma_to_eta(-imag(canonize(evalM, merge(p, (ω=0.0,)))[1]))
        return set_autoptr_eta(alg, η)
    end
    # update the value of eta given the actual parameters
    @eval function reset_autoptr_eta(alg::AutoPTR, cacheval, f::Integrand{typeof($f)}, dom, p)
        η = im_sigma_to_eta(-imag(p[1])) # (estimated) eta from self energy
        return reset_autoptr_eta(alg, cacheval, dom, η)
    end
end

set_autoptr_eta(alg, _, _) = alg
function set_autoptr_eta(alg::AutoPTR, a)
    a >= alg.a && return alg # reuse the existing rule if it has smaller eta
    return AutoPTR(alg.norm, a, alg.nmin, alg.nmax, alg.n₀, alg.Δn, alg.keepmost)
end

reset_autoptr_eta(alg, cacheval, _, _, _) = alg, cacheval
function reset_autoptr_eta(alg::AutoPTR, cacheval, dom, a)
    (r = cacheval.rule) isa AutoBZCore.FourierMonkhorstPackRule || throw(ArgumentError("unexpected AutoPTR rule"))
    # reuse the existing rule if it has smaller eta
    dn = AutoSymPTR.nextnpt(a, alg.nmin, alg.nmax, alg.Δn) <= r.m.Δn && return alg, cacheval
    new_alg = set_autoptr_eta(alg, a) # the alg from set__eta was only used to make the cache and wasn't stored in the solver
    new_rule = AutoBZCore.FourierMonkhorstPackRule(r.s, r.m.syms, a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
    # NOTE that the previous cache may have been created with a different eta, so we replace
    @warn "found Δn=$(dn) larger than original estimate $(r.m.Δn). If you see this a lot you might want to set eta smaller to avoid recomputing PTR rules"
    # it would probably be wise to check npt/dim of the existing rules in the cache and to
    # reuse any that are sufficiently refined for our purposes, but as this may not be the
    # general case when the original estimate is good, I am replacing everything
    resize!(cacheval.cache, 1)
    cacheval.cache[1] = new_rule(eltype(dom), Val(ndims(dom)))
    return new_alg, (rule=new_rule, cache=cacheval.cache, buffer=cacheval.buffer)
end

# helper functions for equispace updates

im_sigma_to_eta(x::UniformScaling) = -x.λ
im_sigma_to_eta(x::Diagonal) = -min(0, maximum(x.diag)) # is this right?
im_sigma_to_eta(x::AbstractMatrix) = im_sigma_to_eta(Diagonal(x)) # is this right?
# instead of imag(Σ) I think we should consider the skew-Hermitian part and
# its spectrum, however even that doesn't actually help because the poles are
# located at det(ω - H(k) - Σ(ω)) = 0. When H and Σ are not simultaneously
# diagonalized, inv(ω - H(k) - Σ(ω)) is no longer a sum of simple rational
# functions, so

# transport and conductivity integrands

function transport_function_integrand((h, vs)::Tuple{Eigen,SVector{N,T}}, β, μ) where {N,T}
    f′ = Diagonal(β .* fermi′.(β .* (h.values .- μ)))
    f′vs = map(v -> f′*v, vs)
    return tr_kron(vs, f′vs)
end

"""
    TransportFunctionIntegrand(hv; β, μ=0)

Computes the following integral
```math
\\D_{\\alpha\\beta} = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
"""
function TransportFunctionIntegrand(hv::AbstractVelocityInterp, args...; kwargs...)
    @assert gauge(hv) isa Hamiltonian
    # TODO change to the Hamiltonian gauge automatically
    return Integrand(transport_function_integrand, hv, args...; kwargs...)
end

tf_params(β, μ)   = promote(β, μ)
tf_params(β; μ=0) = tf_params(β, μ)

function remake_autobz_problem(::typeof(transport_function_integrand), prob)
    return remake(prob, p=canonize(tf_params, prob.p))
end

const TransportFunctionIntegrandType = Integrand{typeof(transport_function_integrand)}

SymRep(D::TransportFunctionIntegrandType) = coord_to_rep(D.s)

# function npt_update(f::TransportFunctionIntegrandType, npt)
#     τ = inv(f.p[1]) # use the scale of exponential localization as η
#     return eta_npt_update(npt, τ, maximum(period(f.s)))
# end

function transport_distribution_integrand_(vs::SVector{N,V}, Aω₁::A, Aω₂::A) where {N,V,A}
    vsAω₁ = map(v -> v * Aω₁, vs)
    vsAω₂ = map(v -> v * Aω₂, vs)
    return tr_kron(vsAω₁, vsAω₂)
end
function transport_distribution_integrand_(vs::SVector{N,V}, Aω::A) where {N,V,A}
    vsAω = map(v -> v * Aω, vs)
    return tr_kron(vsAω, vsAω)
end

spectral_function(G::AbstractMatrix) = (G - G')/(-2pi*im)   # skew-Hermitian part
spectral_function(G::Union{Number,Diagonal}) = imag(G)/(-pi)# optimization
spectral_function(h, M) = spectral_function(gloc_integrand(h, M))

function transport_distribution_integrand((h, vs), Mω₁, Mω₂, isdistinct)
    if isdistinct
        Aω₁ = spectral_function(h, Mω₁)
        Aω₂ = spectral_function(h, Mω₂)
        return transport_distribution_integrand_(vs, Aω₁, Aω₂)
    else
        Aω = spectral_function(h, Mω₁)
        return transport_distribution_integrand_(vs, Aω)
    end
end

"""
    TransportDistributionIntegrand(hv, Σ, ω₁, ω₂, μ)
    TransportDistributionIntegrand(hv, Σ, ω₁, ω₂; μ)
    TransportDistributionIntegrand(hv, Σ; ω₁, ω₂, μ)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See `Integrand` for more details.
"""
function TransportDistributionIntegrand(hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return Integrand(transport_distribution_integrand, hv, Σ, args...; kwargs...)
end

evalM2(Mω₁, Mω₂) = (Mω₁, Mω₂, Mω₁ == Mω₂)
function evalM2(Σ, ω₁, ω₂, μ)
    M = evalM(Σ, ω₁, μ)
    if ω₁ == ω₂
        (M, M, false)
    else
        (M, evalM(Σ, ω₂, μ), true)
    end
end
evalM2(Σ, ω₁, ω₂; μ=0) = evalM2(Σ, ω₁, ω₂, μ)
evalM2(Σ; ω₁, ω₂, μ=0) = evalM2(Σ, ω₁, ω₂, μ)

function remake_autobz_problem(::typeof(transport_distribution_integrand), prob)
    return remake(prob, canonize(evalM2, prob.p))
end

const TransportDistributionIntegrandType = Integrand{typeof(transport_distribution_integrand)}

SymRep(Γ::TransportDistributionIntegrandType) = coord_to_rep(Γ.s)

# function npt_update(Γ::TransportDistributionIntegrandType, npt::Integer)
#     ηω₁ = im_sigma_to_eta(-imag(Γ.p[1]))
#     ηω₂ = im_sigma_to_eta(-imag(Γ.p[2]))
#     return eta_npt_update(npt, min(ηω₁, ηω₂), maximum(period(Γ.s)))
# end


function transport_fermi_integrand(ω, Γ, n, β, Ω)
    return (ω*β)^n * fermi_window(β, ω, Ω) * Γ
end
function transport_fermi_integrand(ω, Γ, n, β, Ω, μ)
    return transport_fermi_integrand(ω, Γ(ω, ω+Ω, μ), n, β, Ω)
end
function transport_fermi_integrand(ω, Σ, n, β, Ω, μ, hv_k)
    Γ = transport_distribution_integrand(hv_k, evalM2(Σ, ω, ω+Ω, μ)...)
    return transport_fermi_integrand(ω, Γ, n, β, Ω)
end
function kinetic_coefficient_integrand(hv_k, frequency_solver)
    return frequency_solver(hv_k)
end

"""
    KineticCoefficientIntegrand([bz=FullBZ,] alg::AutoBZAlgorithm, hv::AbstracVelocity, Σ; n, β, Ω, abstol, reltol, maxiters)
    KineticCoefficientIntegrand([lb=lb(Σ), ub=ub(Σ),] alg, hv::AbstracVelocity, Σ; n, β, Ω, abstol, reltol, maxiters)

A function whose integral over the BZ gives the kinetic
coefficient. Mathematically, this computes
```math
A_{n,\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega (\\beta\\omega)^{n} \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
The argument `alg` determines what the order of integration is. Given a BZ
algorithm, the inner integral is the BZ integral. Otherwise it is the frequency
integral.
"""
function KineticCoefficientIntegrand(bz, alg::AutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    # put the frequency integral outside if the provided algorithm is for the BZ
    transport_integrand = TransportDistributionIntegrand(hv, Σ)
    transport_solver = IntegralSolver(transport_integrand, bz, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    transport_solver(max(-10.0, lb(Σ)), min(10.0, ub(Σ)), 0) # precompile the solver
    return Integrand(transport_fermi_integrand, transport_solver, args...; kwargs...)
end
function KineticCoefficientIntegrand(alg::AutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return KineticCoefficientIntegrand(FullBZ(2pi*I(ndims(hv))), alg, hv, Σ, args...; kwargs...)
end

function KineticCoefficientIntegrand(lb_, ub_, alg, hv::AbstractVelocityInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    # put the frequency integral inside otherwise
    frequency_integrand = Integrand(transport_fermi_integrand, Σ)
    frequency_solver = IntegralSolver(frequency_integrand, lb_, ub_, alg; do_inf_transformation=Val(false), abstol=abstol, reltol=reltol, maxiters=maxiters)
    frequency_solver(hv(fill(0.0, ndims(hv))), 0, 1e100, 0.0, 0) # precompile the solver
    return Integrand(kinetic_coefficient_integrand, hv, frequency_solver, args...; kwargs...)
end
function KineticCoefficientIntegrand(alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return KineticCoefficientIntegrand(lb(Σ), ub(Σ), alg, hv, Σ, args...; kwargs...)
end

kc_params(solver::IntegralSolver, n, β, Ω, μ)   = (solver, n, β, Ω, μ)
kc_params(solver::IntegralSolver, n, β, Ω; μ=0) = (solver, n, β, Ω, μ)
kc_params(solver::IntegralSolver, n, β; Ω, μ=0) = (solver, n, β, Ω, μ)
kc_params(solver::IntegralSolver, n; β, Ω, μ=0) = (solver, n, β, Ω, μ)
kc_params(solver::IntegralSolver; n, β, Ω, μ=0) = (solver, n, β, Ω, μ)


"""
    get_safe_fermi_window_limits(Ω, β, lb, ub)

Given a frequency, `Ω`, inverse temperature, `β`,  returns an interval `(l,u)`
with possibly truncated limits of integration for the frequency integral at each
`(Ω, β)` point that are determined by the [`fermi_window_limits`](@ref) routine
set to the default tolerances for the decay of the Fermi window function. The
arguments `lb` and `ub` are lower and upper limits on the frequency to which the
default result gets truncated if the default result would recommend a wider
interval. If there is any truncation, a warning is emitted to the user, but the
program will continue with the truncated limits.
"""
function get_safe_fermi_window_limits(Ω, β, lb, ub; kwargs...)
    l, u = fermi_window_limits(Ω, β; kwargs...)
    if l < lb
        @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from below"
        l = oftype(l, lb)
    end
    if u+Ω > ub
        @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from above"
        u = oftype(u, ub-Ω)
    end
    l, u
end

# provide safe limits of integration for frequency integrals
function remake_autobz_problem(::typeof(transport_fermi_integrand), lb, ub, p)
    q = canonize(kc_params, p)
    Ω = q[4]; β = q[3]
    iszero(Ω) && isinf(β) && throw(ArgumentError("Ω=0, T=0 not yet implemented. As a workaround, evaluate the KCIntegrand at ω=0"))
    a, b = get_safe_fermi_window_limits(Ω, β, lb, ub)
    return a, b, q
end


# TODO, if T=Ω=0, intercept this stage and replace with distributional integrand
# which would likely cause a type instability
function kc_params(solver_::IntegralSolver{iip,<:Integrand{typeof(kinetic_coefficient_integrand)}}, n_, β_, Ω_, μ_; n=n_, β=β_, Ω=Ω_, μ=μ_) where iip
    iszero(Ω) && isinf(β) && throw(ArgumentError("Ω=0, T=0 not yet implemented. As a workaround, evaluate the KCIntegrand at ω=0"))
    a, b = get_safe_fermi_window_limits(Ω, β, solver_.lb, solver_.ub)
    solver = IntegralSolver(solver_.f, a, b, solver_.alg, sensealg = solver_.sensealg,
            do_inf_transformation = solver_.do_inf_transformation, kwargs = solver_.kwargs,
            abstol = solver_.abstol, reltol = solver_.reltol, maxiters = solver_.maxiters)
    return (solver, n, β, Ω, μ)
end
function remake_autobz_problem(::typeof(kinetic_coefficient_integrand), prob)
    solver = p[1]
    # construct_problem(p[1],
    IntegralSolver
    return prob
end

# const KineticCoefficientIntegrandType = Integrand{typeof(kinetic_coefficient_frequency_integral)}


# SymRep(kc::KineticCoefficientIntegrandType) = coord_to_rep(kc.s)

# function npt_update(f::KineticCoefficientIntegrandType, npt)
# end

"""
    OpticalConductivityIntegrand

Returns a `KineticCoefficientIntegrand` with `n=0`. See
[`KineticCoefficientIntegrand`](@ref) for further details
"""
OpticalConductivityIntegrand(alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    KineticCoefficientIntegrand(alg, hv, Σ, 0, args...; kwargs...)
OpticalConductivityIntegrand(bz, alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    KineticCoefficientIntegrand(bz, alg, hv, Σ, 0, args...; kwargs...)
OpticalConductivityIntegrand(lb, ub, alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    KineticCoefficientIntegrand(lb, ub, alg, hv, Σ, 0, args...; kwargs...)


dos_fermi_integrand(ω, dos, β, μ) =
    fermi(β, ω)*dos(ω, μ)

electron_density_integrand(ω, Σ, h_k, β, μ) =
    fermi(β, ω)*dos_integrand(h_k, evalM(Σ, ω, μ))

electron_density_frequency_integral(h_k, frequency_solver, β, μ) =
    frequency_solver(h_k, β, μ)

"""
    ElectronDensityIntegrand([bz=FullBZ], alg::AutoBZAlgorithm, h::HamiltonianInterp, Σ; β, [μ=0])
    ElectronDensityIntegrand([lb=lb(Σ), ub=ub(Σ),] alg, h::HamiltonianInterp, Σ; β, [μ=0])

A function whose integral over the BZ gives the electron density.
Mathematically, this computes
```math
n(\\mu) = \\int_{-\\infty}^{\\infty} d \\omega f(\\omega) \\operatorname{DOS}(\\omega+\\mu)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
The argument `alg` determines what the order of integration is. Given a BZ
algorithm, the inner integral is the BZ integral. Otherwise it is the frequency
integral.
"""
function ElectronDensityIntegrand(bz, alg::AutoBZAlgorithm, h::HamiltonianInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    dos_int = DOSIntegrand(h, Σ)
    dos_solver = IntegralSolver(dos_int, bz, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    dos_solver(0.0, 0) # precompile the solver
    Integrand(dos_fermi_integrand, dos_solver, args...; kwargs...)
end
ElectronDensityIntegrand(alg::AutoBZAlgorithm, h::HamiltonianInterp, Σ, args...; kwargs...) =
    ElectronDensityIntegrand(FullBZ(2pi*I(ndims(h))), alg, h, Σ, args...; kwargs...)

function ElectronDensityIntegrand(lb, ub, alg, h::HamiltonianInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    frequency_integrand = Integrand(electron_density_integrand, Σ)
    frequency_solver = IntegralSolver(frequency_integrand, lb, ub, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    frequency_solver(h(fill(0.0, ndims(h))), 1.0, 0) # precompile the solver
    Integrand(electron_density_frequency_integral, h, frequency_solver, args...; kwargs...)
end
ElectronDensityIntegrand(alg, h::HamiltonianInterp, Σ, args...; kwargs...) =
    ElectronDensityIntegrand(lb(Σ), ub(Σ), alg, h, Σ, args...; kwargs...)

canonize_density_params(solver::IntegralSolver, β_, μ_; β=β_, μ=μ_) = (solver, β, μ)
function canonize_density_params(solver_::IntegralSolver{iip,<:Integrand{typeof(electron_density_integrand)}}, β_, μ_; β=β_, μ=μ_) where iip
    Σ = solver_.f.p[1]
    a, b = get_safe_fermi_function_limits(β, lb(Σ), ub(Σ))
    # throw((β, μ))
    solver = IntegralSolver(solver_.f, a, b, solver_.alg, sensealg = solver_.sensealg,
            do_inf_transformation = solver_.do_inf_transformation, kwargs = solver_.kwargs,
            abstol = solver_.abstol, reltol = solver_.reltol, maxiters = solver_.maxiters)
    (solver, β, μ)
end
canonize_density_params(solver::IntegralSolver, β; μ=0) = canonize_density_params(solver, β, μ)
canonize_density_params(solver::IntegralSolver; β, μ=0) = canonize_density_params(solver, β, μ)

canonize_density_params(p::MixedParameters) =
    MixedParameters(canonize_density_params(getfield(p, :args)...; getfield(p, :kwargs)...), NamedTuple())
# Integrand(f::typeof(dos_fermi_integrand), p::CanonizeDensityType) =
#     Integrand(f, canonize_density_params(p))
# Integrand(f::typeof(electron_density_frequency_integral), hv::HamiltonianInterp, p::CanonizeDensityType) =
#     Integrand(f, hv, canonize_density_params(p))

function get_safe_fermi_function_limits(β, lb, ub; kwargs...)
    l, u = fermi_function_limits(β; kwargs...)
    if l < lb
        @warn "At β=$β, the interpolant limits the desired frequency window from below"
        l = oftype(l, lb)
    end
    if u > ub
        @warn "At β=$β, the interpolant limits the desired frequency window from above"
        u = oftype(u, ub)
    end
    l, u
end

# provide safe limits of integration for frequency integrals
function construct_problem(s::IntegralSolver{iip,<:Integrand{typeof(dos_fermi_integrand)}}, p::MixedParameters) where iip
    β = canonize_density_params(merge(s.f.p, p))[2]
    a, b = get_safe_fermi_function_limits(β, s.lb, s.ub)
    IntegralProblem{iip}(s.f, a, b, p; s.kwargs...)
end

function aux_transport_distribution_integrand_(vs::SVector{N,V}, Gω₁::G, Gω₂::G) where {N,V,G}
    vsGω₁ = map(v -> v * Gω₁, vs)
    vsGω₂ = map(v -> v * Gω₂, vs)
    Aω₁ = spectral_function(Gω₁)
    Aω₂ = spectral_function(Gω₂)
    vsAω₁ = map(v -> v * Aω₁, vs)
    vsAω₂ = map(v -> v * Aω₂, vs)
    return AuxValue(tr_kron(vsAω₁, vsAω₂), tr_kron(vsGω₁, vsGω₂))
end

aux_transport_distribution_integrand((h, vs), Mω₁, Mω₂) =
    aux_transport_distribution_integrand_(vs, gloc_integrand(h, Mω₁), gloc_integrand(h, Mω₂))

"""
    TransportDistributionIntegrand(hv, Σ, ω₁, ω₂, μ)
    TransportDistributionIntegrand(hv, Σ, ω₁, ω₂; μ)
    TransportDistributionIntegrand(hv, Σ; ω₁, ω₂, μ)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See `Integrand` for more details.
"""
AuxTransportDistributionIntegrand(hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    Integrand(aux_transport_distribution_integrand, hv, Σ, args...; kwargs...)

# function Integrand(f::typeof(aux_transport_distribution_integrand), hv::AbstractVelocityInterp, p::EvalM2Type)
#     Integrand(f, hv, evalM2(p))
# end

const AuxTransportDistributionIntegrandType = Integrand{typeof(aux_transport_distribution_integrand)}

SymRep(Γ::AuxTransportDistributionIntegrandType) = coord_to_rep(Γ.s)

# function npt_update(Γ::AuxTransportDistributionIntegrandType, npt::Integer)
#     ηω₁ = im_sigma_to_eta(-imag(Γ.p[1]))
#     ηω₂ = im_sigma_to_eta(-imag(Γ.p[2]))
#     eta_npt_update(npt, min(ηω₁, ηω₂), maximum(period(Γ.s)))
# end



aux_kinetic_coefficient_integrand(ω, Σ, hv_k, n, β, Ω, μ) =
    (ω*β)^n * fermi_window(β, ω, Ω) * aux_transport_distribution_integrand(hv_k, evalM2(Σ, ω, ω+Ω, μ)...)

function AuxKineticCoefficientIntegrand(bz, alg::AutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    # put the frequency integral outside if the provided algorithm is for the BZ
    transport_integrand = AuxTransportDistributionIntegrand(hv, Σ)
    transport_solver = IntegralSolver(transport_integrand, bz, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    transport_solver(max(-10.0, lb(Σ)), min(10.0, ub(Σ)), 0) # precompile the solver
    Integrand(transport_fermi_integrand, transport_solver, args...; kwargs...)
end
AuxKineticCoefficientIntegrand(alg::AutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    AuxKineticCoefficientIntegrand(FullBZ(2pi*I(ndims(hv))), alg, hv, Σ, args...; kwargs...)

function AuxKineticCoefficientIntegrand(lb_, ub_, alg, hv::AbstractVelocityInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    # put the frequency integral inside otherwise
    frequency_integrand = Integrand(aux_kinetic_coefficient_integrand, Σ)
    frequency_solver = IntegralSolver(frequency_integrand, lb_, ub_, alg; do_inf_transformation=Val(false), abstol=abstol, reltol=reltol, maxiters=maxiters)
    frequency_solver(hv(fill(0.0, ndims(hv))), 0, 1e100, 0.0, 0) # precompile the solver
    Integrand(kinetic_coefficient_frequency_integral, hv, frequency_solver, args...; kwargs...)
end
AuxKineticCoefficientIntegrand(alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    AuxKineticCoefficientIntegrand(lb(Σ), ub(Σ), alg, hv, Σ, args...; kwargs...)

"""
    AuxOpticalConductivityIntegrand

Returns a `AuxKineticCoefficientIntegrand` with `n=0`. See
[`AuxKineticCoefficientIntegrand`](@ref) for further details
"""
AuxOpticalConductivityIntegrand(alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    AuxKineticCoefficientIntegrand(alg, hv, Σ, 0, args...; kwargs...)
AuxOpticalConductivityIntegrand(bz, alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    AuxKineticCoefficientIntegrand(bz, alg, hv, Σ, 0, args...; kwargs...)
AuxOpticalConductivityIntegrand(lb, ub, alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    AuxKineticCoefficientIntegrand(lb, ub, alg, hv, Σ, 0, args...; kwargs...)

function canonize_kc_params(solver_::IntegralSolver{iip,<:Integrand{typeof(aux_kinetic_coefficient_integrand)}}, n_, β_, Ω_, μ_; n=n_, β=β_, Ω=Ω_, μ=μ_) where iip
    iszero(Ω) && isinf(β) && throw(ArgumentError("Ω=0, T=0 not yet implemented. As a workaround, evaluate the KCIntegrand at ω=0"))
    a, b = get_safe_fermi_window_limits(Ω, β, solver_.lb, solver_.ub)
    solver = IntegralSolver(solver_.f, a, b, solver_.alg, sensealg = solver_.sensealg,
            do_inf_transformation = solver_.do_inf_transformation, kwargs = solver_.kwargs,
            abstol = solver_.abstol, reltol = solver_.reltol, maxiters = solver_.maxiters)
    (solver, n, β, Ω, μ)
end

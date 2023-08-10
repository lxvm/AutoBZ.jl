# The integrands here all define a set of canonical parameters. Think of these as default
# arguments, except that we typically don't want to provide defaults in order to require
# that the user provide the necessary data for the problem

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
See [`TrGlocIntegrand`](@ref).
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

    @eval begin
        # define a method to evaluate the Green's function
        $f(h, M) = $f(propagator_denominator(h, M))

        # Define the type alias to have the same behavior as the function
        """
        $($T)(h, Σ, ω, μ)
        $($T)(h, Σ, ω; μ=0)
        $($T)(h, Σ; ω, μ=0)

        Green's function integrands accepting a self energy Σ that can either be a matrix or a
        function of ω (see the self energy section of the documentation for examples)
        """
        function $T(h::AbstractHamiltonianInterp, Σ, args...; kwargs...)
            return FourierIntegrand($f, h, Σ, args...; kwargs...)
        end

        # We provide the following functions to build a cache during construction of IntegralSolvers
        function AutoBZCore.init_solver_cacheval(f::FourierIntegrand{typeof($f)}, dom, alg)
            new_alg = set_autoptr_eta(alg, f, f.p)
            return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), new_alg)
        end

        # evaluate the integrand once for the expected return type
        function (f::FourierIntegrand{typeof($f)})(x, ::CanonicalParameters)
            return typeof(FourierIntegrand(f.f, f.s)(x, canonize(evalM, merge(f.p, (ω=0.0,)))))
        end

        function AutoBZCore.remake_integrand_cache(f::FourierIntegrand{typeof($f)}, dom, p, alg, cacheval, kwargs)
            # pre-evaluate the self energy when remaking the cache
            new_p = canonize(evalM, p)
            # Define default equispace grid stepping (turned off now for simplicity)
            # new_alg, new_cacheval = reset_autoptr_eta(alg, cacheval, f, dom, new_p)
            return AutoBZCore.IntegralCache(f, dom, new_p, alg, cacheval, kwargs)
        end

        # estimate a value of eta that should suffice for most parameters
        # ideally we would have some upper bound on the gradient of the Hamiltonian, v,
        # divide by the largest period, T, and take a = η*v/T
        function set_autoptr_eta(alg::AutoPTR, f::FourierIntegrand{typeof($f)}, p)
            # (estimated) eta from self energy evaluated at the Fermi energy
            η = im_sigma_to_eta(-imag(canonize(evalM, merge(p, (ω=0.0,)))[1]))
            return set_autoptr_eta(alg, η)
        end
        # update the value of eta given the actual parameters
        function reset_autoptr_eta(alg::AutoPTR, cacheval, f::FourierIntegrand{typeof($f)}, dom, p)
            η = im_sigma_to_eta(-imag(p[1])) # (estimated) eta from self energy
            return reset_autoptr_eta(alg, cacheval, dom, η)
        end
    end
end

set_autoptr_eta(alg, _, _) = alg
function set_autoptr_eta(alg::AutoPTR, a)
    a >= alg.a && return alg # reuse the existing rule if it has smaller eta
    return AutoPTR(alg.norm, a, alg.nmin, alg.nmax, alg.n₀, alg.Δn, alg.keepmost, alg.parallel)
end

reset_autoptr_eta(alg, cacheval, _, _, _) = alg, cacheval
function reset_autoptr_eta(alg::AutoPTR, cacheval, dom, a)
    (r = cacheval.rule) isa AutoBZCore.FourierMonkhorstPackRule || throw(ArgumentError("unexpected AutoPTR rule"))
    # reuse the existing rule if it has smaller eta
    (dn = AutoSymPTR.nextnpt(a, alg.nmin, alg.nmax, alg.Δn)) <= r.m.Δn && return alg, cacheval
    new_alg = set_autoptr_eta(alg, a) # the alg from set__eta was only used to make the cache and wasn't stored in the solver
    new_rule = AutoBZCore.FourierMonkhorstPackRule(r.s, r.m.syms, a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
    # NOTE that the previous cache may have been created with a different eta, so we replace
    @warn "found Δn=$(dn) larger than original estimate $(r.m.Δn), but continuing to use the same rule. If you see this a lot you might want to set eta smaller (e.g. $a) to avoid recomputing PTR rules"
    return alg, cacheval
    # it would probably be wise to check npt/dim of the existing rules in the cache and to
    # reuse any that are sufficiently refined for our purposes, but as this may not be the
    # general case when the original estimate is good, I am replacing everything
    # resize!(cacheval.cache, 1)
    # cacheval.cache[1] = new_rule(eltype(dom), Val(ndims(dom)))
    # return new_alg, (rule=new_rule, cache=cacheval.cache, buffer=cacheval.buffer)
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

## transport function

function transport_function_integrand((h, vs)::Tuple{Eigen,SVector{N,T}}, β, μ) where {N,T}
    f′ = Diagonal(β .* fermi′.(β .* (h.values .- μ)))
    f′vs = map(v -> f′*v, vs)
    return tr_kron(vs, f′vs)
end
function transport_function_integrand(v::FourierValue, args...)
    return transport_function_integrand(v.s, args...)
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
    return FourierIntegrand(transport_function_integrand, hv, args...; kwargs...)
end

tf_params(β, μ)   = promote(β, μ)
tf_params(β; μ=0) = tf_params(β, μ)
tf_params(; β, μ=0) = tf_params(β, μ)

const TransportFunctionIntegrandType = FourierIntegrand{typeof(transport_function_integrand)}

function AutoBZCore.init_solver_cacheval(f::TransportFunctionIntegrandType, dom, alg)
    new_alg = set_autoptr_eta(alg, f, f.p)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), new_alg)
end

function (f::TransportFunctionIntegrandType)(x, ::CanonicalParameters)
    return typeof(Integrand(f.f)(x, canonize(tf_params, MixedParameters(1.0, 0.0))))
end

function AutoBZCore.remake_integrand_cache(f::TransportFunctionIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = canonize(tf_params, p)
    # Define default equispace grid stepping
    new_alg, new_cacheval = reset_autoptr_eta(alg, cacheval, f, dom, new_p)
    return AutoBZCore.IntegralCache(f, dom, new_p, new_alg, new_cacheval, kwargs)
end

function set_autoptr_eta(alg::AutoPTR, f::TransportFunctionIntegrandType, p)
    # T=inv(β) is the localization scale, but until β is provided we guess it is one
    β = canonize(tf_params, merge(p, (β=1.0,)))[1]
    return set_autoptr_eta(alg, inv(β))
end
# update the value of eta given the actual parameters
function reset_autoptr_eta(alg::AutoPTR, cacheval, f::TransportFunctionIntegrandType, dom, p)
    T = inv(p[1])
    return reset_autoptr_eta(alg, cacheval, dom, T)
end

SymRep(D::TransportFunctionIntegrandType) = coord_to_rep(D.s)

## transport distribution

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
function transport_distribution_integrand(v::FourierValue, args...)
    return transport_distribution_integrand(v.s, args...)
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
    return FourierIntegrand(transport_distribution_integrand, hv, Σ, args...; kwargs...)
end

# evalM2(Mω₁, Mω₂) = (Mω₁, Mω₂, Mω₁ == Mω₂)
function evalM2(Σ, ω₁, ω₂, μ)
    M = evalM(Σ, ω₁, μ)[1]
    if ω₁ == ω₂
        (M, M, false)
    else
        (M, evalM(Σ, ω₂, μ)[1], true)
    end
end
evalM2(Σ, ω₁, ω₂; μ=0) = evalM2(Σ, ω₁, ω₂, μ)
evalM2(Σ; ω₁, ω₂, μ=0) = evalM2(Σ, ω₁, ω₂, μ)

const TransportDistributionIntegrandType = FourierIntegrand{typeof(transport_distribution_integrand)}

function AutoBZCore.init_solver_cacheval(f::TransportDistributionIntegrandType, dom, alg)
    new_alg = set_autoptr_eta(alg, f, f.p)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), new_alg)
end

function (f::TransportDistributionIntegrandType)(x, ::CanonicalParameters)
    return typeof(FourierIntegrand(f.f, f.s)(x, canonize(evalM2, merge(f.p, (ω₁=0.0, ω₂=0.0)))))
end

function AutoBZCore.remake_integrand_cache(f::TransportDistributionIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = canonize(evalM2, p)
    # Define default equispace grid stepping
    # new_alg, new_cacheval = reset_autoptr_eta(alg, cacheval, f, dom, new_p)
    return AutoBZCore.IntegralCache(f, dom, new_p, alg, cacheval, kwargs)
end

function set_autoptr_eta(alg::AutoPTR, f::TransportDistributionIntegrandType, p)
    # (estimated) eta from self energy evaluated at the Fermi energy
    new_p = canonize(evalM2, merge(p, (ω₁=0.0, ω₂=0.0)))
    η₁ = im_sigma_to_eta(-imag(new_p[1]))
    η₂ = im_sigma_to_eta(-imag(new_p[2]))
    return set_autoptr_eta(alg, min(η₁, η₂))
end
# update the value of eta given the actual parameters
function reset_autoptr_eta(alg::AutoPTR, cacheval, f::TransportDistributionIntegrandType, dom, p)
    η₁ = im_sigma_to_eta(-imag(p[1])) # (estimated) eta from self energy
    η₂ = im_sigma_to_eta(-imag(p[2])) # (estimated) eta from self energy
    return reset_autoptr_eta(alg, cacheval, dom, min(η₁, η₂))
end

SymRep(Γ::TransportDistributionIntegrandType) = coord_to_rep(Γ.s)

# For nested integrands, we have to start to worry about whether or not the outer integral
# is parallelized in order to make the inner integrand threadsafe. If there is
# parallelization, then I opt to deepcopy the inner IntegralSolver for each evaluation out
# of an abundance of caution. Some integral solvers may not need to be copied if their
# algorithm allocates no cache, but I let deepcopy figure this out. We could probably limit
# the number of copies to the number of threads, but I am not so worried about the
# parallel performance (we can't let the integration routines copy the integrand because the
# solvers are often wrapped by anonymous functions that can't be copied).



function transport_fermi_integrand_(ω, Γ, n, β, Ω)
    return (ω*β)^n * fermi_window(β, ω, Ω) * Γ
end
function transport_fermi_integrand(ω, ::Val{ispar}, Γ_, n, β, Ω, μ) where ispar
    Γ = ispar ? deepcopy(Γ_) : Γ_
    return transport_fermi_integrand_(ω, Γ(ω, ω+Ω, μ), n, β, Ω)
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
    # transport_solver(max(-10.0, lb(Σ)), min(10.0, ub(Σ)), 0) # precompile the solver
    return ParameterIntegrand(transport_fermi_integrand, transport_solver, args...; kwargs...)
end
function KineticCoefficientIntegrand(alg::AutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return KineticCoefficientIntegrand(FullBZ(2pi*I(ndims(hv))), alg, hv, Σ, args...; kwargs...)
end

function kc_params(ispar, solver, n, β, Ω, μ)
    iszero(Ω) && isinf(β) && throw(ArgumentError("Ω=0, T=0 not yet implemented. As a workaround, change order of integration or evaluate a TransportDistributionIntegrand at ω₁=ω₂=0"))
    return (ispar, solver, n, β, Ω, μ)
end
kc_params(ispar, solver, n, β, Ω; μ=0.0) = kc_params(ispar, solver, n, β, Ω, μ)
kc_params(ispar, solver, n, β; Ω, μ=0.0) = kc_params(ispar, solver, n, β, Ω, μ)
kc_params(ispar, solver, n; β, Ω, μ=0.0) = kc_params(ispar, solver, n, β, Ω, μ)
kc_params(ispar, solver; n, β, Ω, μ=0.0) = kc_params(ispar, solver, n, β, Ω, μ)

const KCFrequencyType = ParameterIntegrand{typeof(transport_fermi_integrand)}

function AutoBZCore.init_solver_cacheval(f::KCFrequencyType, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end

function (f::KCFrequencyType)(x, ::CanonicalParameters)
    p = canonize(kc_params, MixedParameters(Val(false), f.p[1]; n=0, β=1.0, Ω=0.0))
    return typeof(Integrand(f.f)(x, p))
end

function AutoBZCore.remake_integrand_cache(f::KCFrequencyType, dom, p, alg, cacheval, kwargs)
    new_p = canonize(kc_params, merge(is_threaded(alg), p))
    Ω = new_p[5]; β = new_p[4]
    new_dom = get_safe_fermi_window_limits(Ω, β, dom)
    return AutoBZCore.IntegralCache(f, new_dom, new_p, alg, cacheval, kwargs)
end


function transport_fermi_integrand_inside(ω, Σ, _, n, β, Ω, μ, hv_k)
    Γ = transport_distribution_integrand(hv_k, evalM2(Σ, ω, ω+Ω, μ)...)
    return transport_fermi_integrand_(ω, Γ, n, β, Ω)
end
function kinetic_coefficient_integrand(hv_k::FourierValue, ::Val{ispar}, f, n, β, Ω, μ) where {ispar}
    if iszero(Ω) && isinf(β)
        # we pass in β=4 since fermi_window(4,0,0)=1, the weight of the delta
        # function, and also this prevents (0*β)^n from giving NaN when n!=0
        return f.f(zero(Ω), MixedParameters(n, oftype(β, 4.0), Ω, μ, hv_k))
    end
    frequency_solver = ispar ? deepcopy(f) : f
    return frequency_solver(n, β, Ω, μ, hv_k)
end

function KineticCoefficientIntegrand(lb_, ub_, alg, hv::AbstractVelocityInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    # put the frequency integral inside otherwise
    frequency_integrand = ParameterIntegrand(transport_fermi_integrand_inside, Σ, hv(fill(0.0, ndims(hv))))
    frequency_solver = IntegralSolver(frequency_integrand, lb_, ub_, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    # frequency_solver(0, 1e100, 0.0, 0.0, hv(fill(0.0, ndims(hv)))) # precompile the solver
    return FourierIntegrand(kinetic_coefficient_integrand, hv, frequency_solver, args...; kwargs...)
end
function KineticCoefficientIntegrand(alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return KineticCoefficientIntegrand(lb(Σ), ub(Σ), alg, hv, Σ, args...; kwargs...)
end

const KCFrequencyInsideType = ParameterIntegrand{typeof(transport_fermi_integrand_inside)}

function AutoBZCore.init_solver_cacheval(f::KCFrequencyInsideType, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end

function (f::KCFrequencyInsideType)(x, ::CanonicalParameters)
    return typeof(Integrand(f.f)(x, merge(f.p, MixedParameters(0, 1.0, 0.0, 0.0, f.p[2]))))
end


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
function get_safe_fermi_window_limits(Ω, β, dom; kwargs...)
    int = get_safe_fermi_window_limits(Ω, β, AutoBZCore.endpoints(dom)...; kwargs...)
    return AutoBZCore.PuncturedInterval(int)
end

function kc_inner_params(ispar, solver_, n, β, Ω, μ)
    dom = get_safe_fermi_window_limits(Ω, β, solver_.dom)
    solver = IntegralSolver(solver_.f, dom, solver_.alg, solver_.cacheval, solver_.kwargs)
    return (ispar, solver, convert(Int, n), convert(Float64, β), convert(Float64, Ω), convert(Float64, μ))
end

kc_inner_params(ispar, solver, n, β, Ω; μ=0.0) = kc_inner_params(ispar, solver, n, β, Ω, μ)
kc_inner_params(ispar, solver, n, β; Ω, μ=0.0) = kc_inner_params(ispar, solver, n, β, Ω, μ)
kc_inner_params(ispar, solver, n; β, Ω, μ=0.0) = kc_inner_params(ispar, solver, n, β, Ω, μ)
kc_inner_params(ispar, solver; n, β, Ω, μ=0.0) = kc_inner_params(ispar, solver, n, β, Ω, μ)

const KineticCoefficientIntegrandType = FourierIntegrand{typeof(kinetic_coefficient_integrand)}

function AutoBZCore.init_solver_cacheval(f::KineticCoefficientIntegrandType, dom, alg)
    new_alg = set_autoptr_eta(alg, f, f.p)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), new_alg)
end

function (f::KineticCoefficientIntegrandType)(x, ::CanonicalParameters)
    return typeof(FourierIntegrand(f.f, f.s)(x, canonize(kc_inner_params, MixedParameters(Val(false), f.p[1]; n=0, β=1e100, Ω=0.0))))
end

function AutoBZCore.remake_integrand_cache(f::KineticCoefficientIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = canonize(kc_inner_params, merge(is_threaded(alg), p))
    # We omit adapting the equispace grid stepping since we cannot query the
    # frequency-dependence of the self energy within the inner frequency integral
    # new_alg, new_cacheval = reset_autoptr_eta(alg, cacheval, f, dom, new_p)
    return AutoBZCore.IntegralCache(f, dom, new_p, alg, cacheval, kwargs)
end

function set_autoptr_eta(alg::AutoPTR, ::KineticCoefficientIntegrandType, p)
    # (estimated) eta from self energy evaluated at the Fermi energy
    M = canonize(evalM, MixedParameters(p[1].f.p[1]; ω=0.0))[1]
    η = im_sigma_to_eta(-imag(M))
    return set_autoptr_eta(alg, η)
end

SymRep(kc::KineticCoefficientIntegrandType) = coord_to_rep(kc.s)

"""
    OpticalConductivityIntegrand

Returns a `KineticCoefficientIntegrand` with `n=0`. See
[`KineticCoefficientIntegrand`](@ref) for further details
"""
function OpticalConductivityIntegrand(alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return KineticCoefficientIntegrand(alg, hv, Σ, 0, args...; kwargs...)
end
function OpticalConductivityIntegrand(bz, alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return KineticCoefficientIntegrand(bz, alg, hv, Σ, 0, args...; kwargs...)
end
function OpticalConductivityIntegrand(lb, ub, alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return KineticCoefficientIntegrand(lb, ub, alg, hv, Σ, 0, args...; kwargs...)
end

# Electron density

dos_fermi_integrand_(ω, dos, β) = fermi(β, ω)*dos
function dos_fermi_integrand(ω, ::Val{ispar}, dos_, β, μ) where {ispar}
    dos = ispar ? deepcopy(dos_) : dos_
    return dos_fermi_integrand_(ω, dos(ω, μ), β)
end

"""
    ElectronDensityIntegrand([bz=FullBZ], alg::AutoBZAlgorithm, h::AbstractHamiltonianInterp, Σ; β, [μ=0])
    ElectronDensityIntegrand([lb=lb(Σ), ub=ub(Σ),] alg, h::AbstractHamiltonianInterp, Σ; β, [μ=0])

A function whose integral over the BZ gives the electron density.
Mathematically, this computes
```math
n(\\mu) = \\int_{-\\infty}^{\\infty} d \\omega f(\\omega) \\operatorname{DOS}(\\omega+\\mu)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
The argument `alg` determines what the order of integration is. Given a BZ
algorithm, the inner integral is the BZ integral. Otherwise it is the frequency
integral.

To get the density/number of electrons, multiply the result of this integral by `n_sp/det(bz.B)`
"""
function ElectronDensityIntegrand(bz, alg::AutoBZAlgorithm, h::AbstractHamiltonianInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    dos_int = DOSIntegrand(h, Σ)
    dos_solver = IntegralSolver(dos_int, bz, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    # dos_solver(0.0, 0) # precompile the solver
    return ParameterIntegrand(dos_fermi_integrand, dos_solver, args...; kwargs...)
end
function ElectronDensityIntegrand(alg::AutoBZAlgorithm, h::AbstractHamiltonianInterp, Σ, args...; kwargs...)
    return ElectronDensityIntegrand(FullBZ(2pi*I(ndims(h))), alg, h, Σ, args...; kwargs...)
end

dens_params(ispar, solver, β, μ)     = (ispar, solver, β, μ)
dens_params(ispar, solver, β; μ=0.0) = (ispar, solver, β, μ)
dens_params(ispar, solver; β, μ=0.0) = (ispar, solver, β, μ)

const DensityFrequencyType = ParameterIntegrand{typeof(dos_fermi_integrand)}

function AutoBZCore.init_solver_cacheval(f::DensityFrequencyType, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end

function (f::DensityFrequencyType)(x, ::CanonicalParameters)
    return typeof(Integrand(f.f)(x, canonize(dens_params, MixedParameters(Val(false), f.p[1]; β=1.0))))
end

function AutoBZCore.remake_integrand_cache(f::DensityFrequencyType, dom, p, alg, cacheval, kwargs)
    new_p = canonize(dens_params, merge(is_threaded(alg), p))
    β = new_p[3]
    new_dom = get_safe_fermi_function_limits(β, dom)
    return AutoBZCore.IntegralCache(f, new_dom, new_p, alg, cacheval, kwargs)
end


function dos_fermi_integrand_inside(ω, Σ, _, β, μ, h_k)
    return dos_fermi_integrand_(ω, dos_integrand(h_k, evalM(Σ, ω, μ)...), β)
end

function electron_density_integrand(h_k::FourierValue, ::Val{ispar}, f) where {ispar}
    frequency_solver = ispar ? deepcopy(f) : f
    return frequency_solver(h_k)
end

function ElectronDensityIntegrand(lb, ub, alg, h::AbstractHamiltonianInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    frequency_integrand = ParameterIntegrand(dos_fermi_integrand_inside, Σ, h(fill(0.0, ndims(h))))
    frequency_solver = IntegralSolver(frequency_integrand, lb, ub, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    # frequency_solver(h(fill(0.0, ndims(h))), 1.0, 0) # precompile the solver
    return FourierIntegrand(electron_density_integrand, h, frequency_solver, args...; kwargs...)
end
function ElectronDensityIntegrand(alg, h::AbstractHamiltonianInterp, Σ, args...; kwargs...)
    return ElectronDensityIntegrand(lb(Σ), ub(Σ), alg, h, Σ, args...; kwargs...)
end

const DensityFrequencyInsideType = ParameterIntegrand{typeof(dos_fermi_integrand_inside)}

function AutoBZCore.init_solver_cacheval(f::DensityFrequencyInsideType, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end

function (f::DensityFrequencyInsideType)(x, ::CanonicalParameters)
    return typeof(Integrand(f.f)(x, merge(f.p, MixedParameters(1.0, 0.0, f.p[2]))))
end

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
function get_safe_fermi_function_limits(β, dom; kwargs...)
    int = get_safe_fermi_function_limits(β, AutoBZCore.endpoints(dom)...; kwargs...)
    return AutoBZCore.PuncturedInterval(int)
end

function dens_params_inside(ispar, solver_::IntegralSolver, β, μ)
    dom = get_safe_fermi_function_limits(β, solver_.dom)
    g = ParameterIntegrand{typeof(solver_.f.f)}(solver_.f.f, merge(solver_.f.p, (β, μ)))
    solver = IntegralSolver(g, dom, solver_.alg, solver_.cacheval, solver_.kwargs)
    return (ispar, solver)
end
dens_params_inside(ispar, solver, β; μ=0.0) = dens_params_inside(ispar, solver, β, μ)
dens_params_inside(ispar, solver; β, μ=0.0) = dens_params_inside(ispar, solver, β, μ)

const ElectronDensityIntegrandType = FourierIntegrand{typeof(electron_density_integrand)}

function AutoBZCore.init_solver_cacheval(f::ElectronDensityIntegrandType, dom, alg)
    new_alg = set_autoptr_eta(alg, f, f.p)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), new_alg)
end

function (f::ElectronDensityIntegrandType)(x, ::CanonicalParameters)
    p = MixedParameters(Val(false), f.p[1]; β=1e100)
    return typeof(FourierIntegrand(f.f, f.s)(x, canonize(dens_params_inside, p)))
end

function AutoBZCore.remake_integrand_cache(f::ElectronDensityIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = canonize(dens_params_inside, merge(is_threaded(alg), p))
    # We omit adapting the equispace grid stepping since we cannot query the
    # frequency-dependence of the self energy within the inner frequency integral
    # new_alg, new_cacheval = reset_autoptr_eta(alg, cacheval, f, dom, new_p)
    return AutoBZCore.IntegralCache(f, dom, new_p, alg, cacheval, kwargs)
end

function set_autoptr_eta(alg::AutoPTR, ::ElectronDensityIntegrandType, p)
    # (estimated) eta from self energy evaluated at the Fermi energy
    # if we knew β then we would pick the larger of η and inv(β) since there are no
    # interband transitions
    M = canonize(evalM, MixedParameters(p[1].f.p[1]; ω=0.0))[1]
    η = im_sigma_to_eta(-imag(M))
    return set_autoptr_eta(alg, η)
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
function aux_transport_distribution_integrand_(vs::SVector{N,V}, Gω::G) where {N,V,G}
    vsGω = map(v -> v * Gω, vs)
    Aω = spectral_function(Gω)
    vsAω = map(v -> v * Aω, vs)
    return AuxValue(tr_kron(vsAω, vsAω), tr_kron(vsGω, vsGω))
end

function aux_transport_distribution_integrand((h, vs), Mω₁, Mω₂, isdistinct)
    if isdistinct
        Gω₁ = gloc_integrand(h, Mω₁)
        Gω₂ = gloc_integrand(h, Mω₂)
        return aux_transport_distribution_integrand_(vs, Gω₁, Gω₂)
    else
        Gω = gloc_integrand(h, Mω₁)
        return aux_transport_distribution_integrand_(vs, Gω)
    end
end
function aux_transport_distribution_integrand(x::FourierValue, args...)
    return aux_transport_distribution_integrand(x.s, args...)
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
function AuxTransportDistributionIntegrand(hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return FourierIntegrand(aux_transport_distribution_integrand, hv, Σ, args...; kwargs...)
end

const AuxTransportDistributionIntegrandType = FourierIntegrand{typeof(aux_transport_distribution_integrand)}

function AutoBZCore.init_solver_cacheval(f::AuxTransportDistributionIntegrandType, dom, alg)
    new_alg = set_autoptr_eta(alg, f, f.p)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), new_alg)
end

function (f::AuxTransportDistributionIntegrandType)(x, ::CanonicalParameters)
    return typeof(FourierIntegrand(f.f, f.s)(x, canonize(evalM2, merge(f.p, (ω₁=0.0, ω₂=0.0)))))
end

function AutoBZCore.remake_integrand_cache(f::AuxTransportDistributionIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = canonize(evalM2, p)
    # Define default equispace grid stepping
    # new_alg, new_cacheval = reset_autoptr_eta(alg, cacheval, f, dom, new_p)
    return AutoBZCore.IntegralCache(f, dom, new_p, alg, cacheval, kwargs)
end

function set_autoptr_eta(alg::AutoPTR, f::AuxTransportDistributionIntegrandType, p)
    # (estimated) eta from self energy evaluated at the Fermi energy
    new_p = canonize(evalM2, merge(p, (ω₁=0.0, ω₂=0.0)))
    η₁ = im_sigma_to_eta(-imag(new_p[1]))
    η₂ = im_sigma_to_eta(-imag(new_p[2]))
    return set_autoptr_eta(alg, min(η₁, η₂))
end
# update the value of eta given the actual parameters
function reset_autoptr_eta(alg::AutoPTR, cacheval, f::AuxTransportDistributionIntegrandType, dom, p)
    η₁ = im_sigma_to_eta(-imag(p[1])) # (estimated) eta from self energy
    η₂ = im_sigma_to_eta(-imag(p[2])) # (estimated) eta from self energy
    return reset_autoptr_eta(alg, cacheval, dom, min(η₁, η₂))
end


SymRep(Γ::AuxTransportDistributionIntegrandType) = coord_to_rep(Γ.s)


function aux_kinetic_coefficient_integrand(ω, Σ, hv_k, n, β, Ω, μ)
    Γ = aux_transport_distribution_integrand(hv_k, evalM2(Σ, ω, ω+Ω, μ)...)
    return (ω*β)^n * fermi_window(β, ω, Ω) * Γ
end

function AuxKineticCoefficientIntegrand(bz, alg::AutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    # put the frequency integral outside if the provided algorithm is for the BZ
    transport_integrand = AuxTransportDistributionIntegrand(hv, Σ)
    transport_solver = IntegralSolver(transport_integrand, bz, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    # transport_solver(max(-10.0, lb(Σ)), min(10.0, ub(Σ)), 0) # precompile the solver
    return ParameterIntegrand(transport_fermi_integrand, transport_solver, args...; kwargs...)
end
function AuxKineticCoefficientIntegrand(alg::AutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return AuxKineticCoefficientIntegrand(FullBZ(2pi*I(ndims(hv))), alg, hv, Σ, args...; kwargs...)
end

function aux_transport_fermi_integrand_inside(ω, Σ, _, n, β, Ω, μ, hv_k)
    Γ = aux_transport_distribution_integrand(hv_k, evalM2(Σ, ω, ω+Ω, μ)...)
    return transport_fermi_integrand_(ω, Γ, n, β, Ω)
end

function AuxKineticCoefficientIntegrand(lb_, ub_, alg, hv::AbstractVelocityInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    # put the frequency integral inside otherwise
    frequency_integrand = ParameterIntegrand(aux_transport_fermi_integrand_inside, Σ, hv(fill(0.0, ndims(hv))))
    frequency_solver = IntegralSolver(frequency_integrand, lb_, ub_, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    # frequency_solver(hv(fill(0.0, ndims(hv))), 0, 1e100, 0.0, 0) # precompile the solver
    return FourierIntegrand(kinetic_coefficient_integrand, hv, frequency_solver, args...; kwargs...)
end
function AuxKineticCoefficientIntegrand(alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return AuxKineticCoefficientIntegrand(lb(Σ), ub(Σ), alg, hv, Σ, args...; kwargs...)
end

const AuxKCFrequencyInsideType = ParameterIntegrand{typeof(aux_transport_fermi_integrand_inside)}

function AutoBZCore.init_solver_cacheval(f::AuxKCFrequencyInsideType, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end

function (f::AuxKCFrequencyInsideType)(x, ::CanonicalParameters)
    return typeof(Integrand(f.f)(x, merge(f.p, MixedParameters(0, 1.0, 0.0, 0.0, f.p[2]))))
end

"""
    AuxOpticalConductivityIntegrand

Returns a `AuxKineticCoefficientIntegrand` with `n=0`. See
[`AuxKineticCoefficientIntegrand`](@ref) for further details
"""
function AuxOpticalConductivityIntegrand(alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return AuxKineticCoefficientIntegrand(alg, hv, Σ, 0, args...; kwargs...)
end
function AuxOpticalConductivityIntegrand(bz, alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return AuxKineticCoefficientIntegrand(bz, alg, hv, Σ, 0, args...; kwargs...)
end
function AuxOpticalConductivityIntegrand(lb, ub, alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...)
    return AuxKineticCoefficientIntegrand(lb, ub, alg, hv, Σ, 0, args...; kwargs...)
end

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
gloc_integrand(G) = _inv(G)

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

Returns `-imag(tr(inv(M-h)))/pi` where `M = ω*I-Σ(ω)`. It is unsafe to use
this in the inner integral for small eta due to the localized tails of the
integrand. The default, safe version also integrates the real part which is less
localized, at the expense of a slight slow-down due to complex arithmetic.
See [`TrGlocIntegrand`](@ref).
"""
dos_integrand(G) = -imag(tr_inv(G))/pi

# shift energy by chemical potential, but not self energy
_evalM(Σ::Union{AbstractMatrix,UniformScaling}, ω, μ) = ((ω+μ)*I - Σ,)
_evalM(Σ::AbstractSelfEnergy, ω, μ) = _evalM(Σ(ω), ω, μ)
evalM(; Σ, ω, μ=zero(ω)) = _evalM(Σ, ω, μ)

# given mixed parameters and a function that maps them to a tuple of a canonical
# form for the integrand, return new mixed parameters
function canonize(f, p::MixedParameters)
    params = f(getfield(p, :args)...; getfield(p, :kwargs)...)
    return MixedParameters(params, NamedTuple())
end

function make_fourier_nest(p, f, w)
    if length(w.cache) == 1
        return nothing
    else
        x = period(w.series)
        fx = FourierIntegrand(p, w)(x, CanonicalParameters())
        return make_fourier_nest_(f, fx, w, x...)
    end
end
function make_fourier_nest_(f, fx, w, x1, x...)
    len = length(w.cache)
    if x isa Tuple{}
        if len == 1
            return deepcopy(f)
        else
            f1s = ntuple(n -> deepcopy(f), Val(len))
            y1s = typeof(fx)[]
            x1s = eltype(x1)[]
            return NestedBatchIntegrand(collect(f1s), y1s, x1s, max_batch=10^6)
        end
    else
        nests = ntuple(n -> make_fourier_nest_(f, fx, w.cache[n], x1, x[begin:end-1]...), Val(len))
        if (len = length(w.cache)) == 1
            return nests[1]
        else
            ys = typeof(fx*x1*prod(x[begin:end-1]))[]
            xs = eltype(x[end])[]
            return NestedBatchIntegrand(collect(nests), ys, xs, max_batch=10^6)
        end
    end
end

function alloc_nest_buffers(nest, x)
    nest isa NestedBatchIntegrand || return nest
    xs = eltype(nest.x) === Nothing ? typeof(x[end])[] : nest.x
    workers = map(f -> alloc_nest_buffers(f, x[begin:end-1]), nest.f)
    return NestedBatchIntegrand(workers, nest.y, xs, max_batch = nest.max_batch)
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
            $($T)(h; Σ, ω, μ=0)

        Green's function integrands accepting a self energy Σ that can either be a matrix or a
        function of ω (see the self energy section of the documentation for examples)
        """
        function $T(h::AbstractHamiltonianInterp; kwargs...)
            return FourierIntegrand($f, h; kwargs...)
        end
        function $T(w::FourierWorkspace{<:AbstractHamiltonianInterp}; kwargs...)
            p = ParameterIntegrand($f; kwargs...)
            nest = make_fourier_nest(p, ParameterIntegrand($f), w)
            return nest === nothing ? FourierIntegrand(p, w) : FourierIntegrand(p, w, nest)
        end

        # We provide the following functions to build a cache during construction of IntegralSolvers
        function AutoBZCore.init_solver_cacheval(f::FourierIntegrand{typeof($f)}, dom, alg)
            nest = alloc_nest_buffers(f.nest, interior_point(dom.lims))
            new_f = nest === nothing ? f : FourierIntegrand(f.f, f.w, nest)
            return AutoBZCore.init_cacheval(new_f, dom, CanonicalParameters(), alg)
        end

        # evaluate the integrand once for the expected return type
        function (f::FourierIntegrand{typeof($f)})(x::FourierValue, ::CanonicalParameters)
            el = real(_eltype(x.s))
            canonical_p = (Σ=EtaSelfEnergy(oneunit(el)), ω=zero(el))
            return FourierIntegrand(f.f.f, f.w)(x, canonize(evalM, merge(canonical_p, f.f.p)))
        end

        function AutoBZCore.remake_integrand_cache(f::FourierIntegrand{typeof($f)}, dom, p, alg, cacheval, kwargs)
            # pre-evaluate the self energy when remaking the cache
            new_p = canonize(evalM, p)
            # Define default equispace grid stepping for AutoPTR
            new_alg = choose_autoptr_step(alg, sigma_to_eta(p.Σ), f.w.series)
            new_cacheval = alg == new_alg ? cacheval : AutoBZCore.init_cacheval(f, dom, p, new_alg)
            return AutoBZCore.IntegralCache(f, dom, new_p, new_alg, new_cacheval, kwargs)
        end
    end
end

choose_autoptr_step(alg, _...) = alg

function choose_autoptr_step(alg::AutoPTR, a::Real)
    return a == alg.a ? alg : AutoPTR(a=Float64(a), norm=alg.norm, nmin=alg.nmin, nmax=alg.nmax, n₀=alg.n₀, Δn=alg.Δn, keepmost=alg.keepmost, nthreads=alg.nthreads)
end
# estimate a value of eta that should suffice for most parameters
# ideally we would have some upper bound on the gradient of the Hamiltonian, v,
# divide by the largest period, T, and take a = η/v/T (dimensionless)
function choose_autoptr_step(alg::AutoPTR, η::Number, h::AbstractHamiltonianInterp)
    vT = velocity_bound(h)
    a = freq2rad(η/vT) # 2pi*a′/T as in eqn. 3.24 of Trefethen's "The Exponentially Convergent Trapezoidal rule"
    return choose_autoptr_step(alg, a)
end
# heuristic helper functions for equispace updates

sigma_to_eta(x::UniformScaling) = -imag(x.λ)
function sigma_to_eta(x::Diagonal)
    # is this right?
    imx = imag(x.diag)
    return -min(zero(eltype(imx)), maximum(imx))
end
sigma_to_eta(x::AbstractMatrix) = sigma_to_eta(Diagonal(x)) # is this right?
sigma_to_eta(Σ::AbstractSelfEnergy) = sigma_to_eta(Σ(zero((lb(Σ)+ub(Σ))/2))) # use self energy at Fermi energy
# instead of imag(Σ) I think we should consider the skew-Hermitian part and
# its spectrum, however even that doesn't actually help because the poles are
# located at det(ω - H(k) - Σ(ω)) = 0. When H and Σ are not simultaneously
# diagonalized, inv(ω - H(k) - Σ(ω)) is no longer a sum of simple rational
# functions, so

# returns a bound (L₂ norm) on the velocity times the period v*T
function velocity_bound(f::FourierSeries)
    # use Parseval's theorem to compute ||v||₂ = √ ∫dk/V |v|^2 from v_R and volume V
    # rigorously, we should care about the maximal band velocity, but that is more elaborate
    return sqrt(sum(i -> (hypot(map(*, map(+, i.I, f.o), f.t)...)*norm(f.c[i]))^2, CartesianIndices(f.c)))
end

velocity_bound(h::AbstractHamiltonianInterp) = velocity_bound(parentseries(h))

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
function TransportFunctionIntegrand(hv::AbstractVelocityInterp; kwargs...)
    @assert gauge(hv) isa Hamiltonian
    # TODO change to the Hamiltonian gauge automatically
    return FourierIntegrand(transport_function_integrand, hv; kwargs...)
end

tf_params(; β, μ=zero(inv(oneunit(β)))) = (β, μ)

const TransportFunctionIntegrandType = FourierIntegrand{typeof(transport_function_integrand)}

function AutoBZCore.init_solver_cacheval(f::TransportFunctionIntegrandType, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end

function (f::TransportFunctionIntegrandType)(x::FourierValue, ::CanonicalParameters)
    el = real(_eltype(x.s[1]))
    canonical_p = (β=inv(oneunit(el)), μ=zero(el))
    return FourierIntegrand(f.f.f, f.w)(x, canonize(tf_params, merge(canonical_p, f.f.p)))
end

function AutoBZCore.remake_integrand_cache(f::TransportFunctionIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = canonize(tf_params, p)
    # Define default equispace grid stepping from the localization scale T=inv(β)
    new_alg = choose_autoptr_step(alg, inv(new_p[1]), parentseries(f.w.series))
    new_cacheval = alg == new_alg ? cacheval : AutoBZCore.init_cacheval(f, dom, p, new_alg)
    return AutoBZCore.IntegralCache(f, dom, new_p, new_alg, new_cacheval, kwargs)
end

SymRep(D::TransportFunctionIntegrandType) = coord_to_rep(D.w.series)

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

function spectral_function(G::AbstractMatrix)
    T = real(eltype(G)) # get floating point type from input
    imtwo = complex(zero(one(T)), one(T)+one(T))
    return (G - G')/(-imtwo*pi)   # skew-Hermitian part
end
spectral_function(G::Union{Number,Diagonal}) = -imag(G)/pi # optimization
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
    TransportDistributionIntegrand(hv; Σ, ω₁, ω₂, μ=0)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See `Integrand` for more details.
"""
function TransportDistributionIntegrand(hv::AbstractVelocityInterp; kwargs...)
    return FourierIntegrand(transport_distribution_integrand, hv; kwargs...)
end
function TransportDistributionIntegrand(w::FourierWorkspace{<:AbstractVelocityInterp}; kwargs...)
    p = ParameterIntegrand(transport_distribution_integrand; kwargs...)
    nest = make_fourier_nest(p, ParameterIntegrand(transport_distribution_integrand), w)
    return nest === nothing ? FourierIntegrand(p, w) : FourierIntegrand(p, w, nest)
end

function _evalM2(Σ, ω₁::T, ω₂::T, μ::T) where {T}
    M = _evalM(Σ, ω₁, μ)[1]
    if ω₁ == ω₂
        (M, M, false)
    else
        (M, _evalM(Σ, ω₂, μ)[1], true)
    end
end
evalM2(; Σ, ω₁, ω₂, μ=zero(ω₁), kws...) = _evalM2(Σ, promote(ω₁, ω₂, μ)...)

const TransportDistributionIntegrandType = FourierIntegrand{typeof(transport_distribution_integrand)}

function AutoBZCore.init_solver_cacheval(f::TransportDistributionIntegrandType, dom, alg)
    nest = alloc_nest_buffers(f.nest, interior_point(dom.lims))
    new_f = nest === nothing ? f : FourierIntegrand(f.f, f.w, nest)
    return AutoBZCore.init_cacheval(new_f, dom, CanonicalParameters(), alg)
end

function (f::TransportDistributionIntegrandType)(x::FourierValue, ::CanonicalParameters)
    el = real(_eltype(x.s[1]))
    canonical_p = (Σ=EtaSelfEnergy(oneunit(el)), ω₁=zero(el), ω₂=zero(el))
    return FourierIntegrand(f.f.f, f.w)(x, canonize(evalM2, merge(canonical_p, f.f.p)))
end

function AutoBZCore.remake_integrand_cache(f::TransportDistributionIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = canonize(evalM2, p)
    # Define default equispace grid stepping
    new_alg = choose_autoptr_step(alg, sigma_to_eta(p.Σ), parentseries(f.w.series))
    new_cacheval = alg == new_alg ? cacheval : AutoBZCore.init_cacheval(f, dom, p, new_alg)
    return AutoBZCore.IntegralCache(f, dom, new_p, new_alg, new_cacheval, kwargs)
end

SymRep(Γ::TransportDistributionIntegrandType) = coord_to_rep(Γ.w.series)

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
function transport_fermi_integrand(ω, Γ, Σ, n, β, Ω, μ)
    return transport_fermi_integrand_(ω, Γ(; Σ, ω₁=ω, ω₂=ω+Ω, μ), n, β, Ω)
end

"""
    KineticCoefficientIntegrand(bz, alg::AutoBZAlgorithm, hv::AbstracVelocity; Σ, n, β, Ω, μ=0, abstol, reltol, maxiters)
    KineticCoefficientIntegrand(lb, ub, alg, hv::AbstracVelocity; Σ, n, β, Ω, μ=0, abstol, reltol, maxiters)

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
function KineticCoefficientIntegrand(bz, alg::AutoBZAlgorithm, hv::Union{T,FourierWorkspace{T}}; kwargs...) where {T<:AbstractVelocityInterp}
    solver_kws, kws = nested_solver_kwargs(NamedTuple(kwargs))
    # put the frequency integral outside if the provided algorithm is for the BZ
    transport_integrand = TransportDistributionIntegrand(hv)
    transport_solver = IntegralSolver(transport_integrand, bz, alg; solver_kws...)
    return ParameterIntegrand(transport_fermi_integrand, transport_solver; kws...)
end

function nested_solver_kwargs(___kws::NamedTuple)
    akw, __kws = _peel_abstol(; ___kws...)
    rkw, _kws = _peel_reltol(; __kws...)
    mkw, kws = _peel_maxiters(; _kws...)
    return merge(akw, rkw, mkw), kws
end
_peel_abstol(; abstol=nothing, kws...) = (isnothing(abstol) ? NamedTuple() : (; abstol=abstol)), NamedTuple(kws)
_peel_reltol(; reltol=nothing, kws...) = (isnothing(reltol) ? NamedTuple() : (; reltol=reltol)), NamedTuple(kws)
_peel_maxiters(; maxiters=nothing, kws...) = (isnothing(maxiters) ? NamedTuple() : (; maxiters=maxiters)), NamedTuple(kws)


function kc_params(solver; Σ, n, β, Ω, μ=zero(Ω), kws...)
    iszero(Ω) && isinf(β) && throw(ArgumentError("Ω=0, T=0 not yet implemented. As a workaround, change order of integration or evaluate a TransportDistributionIntegrand at ω₁=ω₂=0"))
    return (solver, Σ, n, β, Ω, μ)
end

const KCFrequencyType = ParameterIntegrand{typeof(transport_fermi_integrand)}

function AutoBZCore.init_solver_cacheval(f::KCFrequencyType, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end

function (f::KCFrequencyType)(ω, ::CanonicalParameters)
    canonical_p = (Σ=EtaSelfEnergy(oneunit(ω)), n=0, β=inv(oneunit(ω)), Ω=zero(ω))
    return ParameterIntegrand(f.f)(ω, canonize(kc_params, merge(canonical_p, f.p)))
end

function AutoBZCore.remake_integrand_cache(f::KCFrequencyType, dom, p, alg, cacheval, kwargs)
    new_p = canonize(kc_params, p)
    Ω = new_p[5]; β = new_p[4]
    new_dom = get_safe_fermi_window_limits(Ω, β, dom)
    return AutoBZCore.IntegralCache(f, new_dom, new_p, alg, cacheval, kwargs)
end


function transport_fermi_integrand_inside(ω; Σ, n, β, Ω, μ, hv_k)
    Γ = transport_distribution_integrand(hv_k, evalM2(; Σ, ω₁=ω, ω₂=ω+Ω, μ)...)
    return transport_fermi_integrand_(ω, Γ, n, β, Ω)
end
function kinetic_coefficient_integrand(hv_k::FourierValue, f, Σ, n, β, Ω, μ)
    if iszero(Ω) && isinf(β)
        # we pass in β=4 since fermi_window(4,0,0)=1, the weight of the delta
        # function, and also this prevents (0*β)^n from giving NaN when n!=0
        return Ω * f.f(Ω, MixedParameters(; Σ, n, β=4*oneunit(β), Ω, μ, hv_k))
    end
    return f(; Σ, n, β, Ω, μ, hv_k)
end

struct KCFrequencyIntegral{T}
    solver::T
end

function (kc::KCFrequencyIntegral)(hv_k, dom, Σ, n, β, Ω, μ)
    solver = IntegralSolver(kc.solver.f, dom, kc.solver.alg, kc.solver.cacheval, kc.solver.kwargs)
    return kinetic_coefficient_integrand(hv_k, solver, Σ, n, β, Ω, μ)
end

function KineticCoefficientIntegrand(lb_, ub_, alg, hv::AbstractVelocityInterp, args...; kwargs...)
    solver_kws, kws = nested_solver_kwargs(NamedTuple(kwargs))
    # put the frequency integral inside otherwise
    frequency_integrand = ParameterIntegrand(transport_fermi_integrand_inside; hv_k=hv(period(hv)))
    frequency_solver = IntegralSolver(frequency_integrand, lb_, ub_, alg; solver_kws...)
    dom = AutoBZCore.PuncturedInterval((lb_, ub_))
    return FourierIntegrand(KCFrequencyIntegral(frequency_solver), hv, dom, args...; kws...)
end

function KineticCoefficientIntegrand(lb_, ub_, alg, w::FourierWorkspace{<:AbstractVelocityInterp}, args...; kwargs...)
    # put the frequency integral inside otherwise
    solver_kws, kws = nested_solver_kwargs(NamedTuple(kwargs))
    frequency_integrand = ParameterIntegrand(transport_fermi_integrand_inside; hv_k=w(period(w.series)))
    dom = AutoBZCore.PuncturedInterval((lb_, ub_))
    frequency_solver = IntegralSolver(frequency_integrand, dom, alg; solver_kws...)
    int = KCFrequencyIntegral(frequency_solver)
    p = ParameterIntegrand(int, dom, args...; kws...)
    nest = make_fourier_nest(p, ParameterIntegrand(int), w)
    return nest === nothing ? FourierIntegrand(p, w) : FourierIntegrand(p, w, nest)
end

const KCFrequencyInsideType = ParameterIntegrand{typeof(transport_fermi_integrand_inside)}

function AutoBZCore.init_solver_cacheval(f::KCFrequencyInsideType, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end

function (f::KCFrequencyInsideType)(ω, ::CanonicalParameters)
    canonical_p = (Σ=EtaSelfEnergy(oneunit(ω)), n=0, β=inv(oneunit(ω)), Ω=zero(ω), μ=zero(ω))
    return ParameterIntegrand(f.f)(ω, merge(canonical_p, f.p))
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

function kc_inner_params(dom, Σ, n, β, Ω, μ; kws...)
    new_dom = get_safe_fermi_window_limits(Ω, β, dom)
    return (new_dom, Σ, n, β, Ω, μ)
end

kc_inner_params(dom, Σ, n, β, Ω; μ=zero(Ω)) = kc_inner_params(dom, Σ, n, β, Ω, μ)
kc_inner_params(dom, Σ, n, β; Ω, μ=zero(Ω)) = kc_inner_params(dom, Σ, n, β, Ω, μ)
kc_inner_params(dom, Σ, n; β, Ω, μ=zero(Ω)) = kc_inner_params(dom, Σ, n, β, Ω, μ)
kc_inner_params(dom, Σ; n, β, Ω, μ=zero(Ω)) = kc_inner_params(dom, Σ, n, β, Ω, μ)
kc_inner_params(dom; Σ, n, β, Ω, μ=zero(Ω)) = kc_inner_params(dom, Σ, n, β, Ω, μ)

const KineticCoefficientIntegrandType = FourierIntegrand{<:KCFrequencyIntegral}

function AutoBZCore.init_solver_cacheval(f::KineticCoefficientIntegrandType, dom, alg)
    nest = alloc_nest_buffers(f.nest, interior_point(dom.lims))
    new_f = nest === nothing ? f : FourierIntegrand(f.f, f.w, nest)
    return AutoBZCore.init_cacheval(new_f, dom, CanonicalParameters(), alg)
end

function (f::KineticCoefficientIntegrandType)(x::FourierValue, ::CanonicalParameters)
    el = real(_eltype(x.s[1]))
    canonical_p = (Σ=EtaSelfEnergy(oneunit(el)), n=0, β=inv(oneunit(el)), Ω=zero(el))
    return FourierIntegrand(f.f.f, f.w)(x, canonize(kc_inner_params, merge(canonical_p, f.f.p)))
end

function AutoBZCore.remake_integrand_cache(f::KineticCoefficientIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = canonize(kc_inner_params, p)
    Σ = new_p[2]
    # Define default equispace grid stepping
    new_alg = choose_autoptr_step(alg, sigma_to_eta(Σ), parentseries(f.w.series))
    new_cacheval = alg == new_alg ? cacheval : AutoBZCore.init_cacheval(f, dom, p, new_alg)
    return AutoBZCore.IntegralCache(f, dom, new_p, new_alg, new_cacheval, kwargs)
end


SymRep(kc::KineticCoefficientIntegrandType) = coord_to_rep(kc.w.series)

"""
    OpticalConductivityIntegrand

Returns a `KineticCoefficientIntegrand` with `n=0`. See
[`KineticCoefficientIntegrand`](@ref) for further details
"""
OpticalConductivityIntegrand(args...; kws...) = KineticCoefficientIntegrand(args...; kws..., n=0)

# Electron density

dos_fermi_integrand_(ω, dos, β) = fermi(β, ω)*dos
function dos_fermi_integrand(ω, dos, Σ, β, μ)
    return dos_fermi_integrand_(ω, dos(; Σ, ω, μ), β)
end

"""
    ElectronDensityIntegrand(bz, alg::AutoBZAlgorithm, h::AbstractHamiltonianInterp; Σ, β, [μ=0])
    ElectronDensityIntegrand(lb, ub, alg, h::AbstractHamiltonianInterp; Σ, β, [μ=0])

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
function ElectronDensityIntegrand(bz, alg::AutoBZAlgorithm, h::Union{T,FourierWorkspace{<:T}}; kwargs...) where {T<:AbstractHamiltonianInterp}
    solver_kws, kws = nested_solver_kwargs(NamedTuple(kwargs))
    dos_int = DOSIntegrand(h)
    dos_solver = IntegralSolver(dos_int, bz, alg; solver_kws...)
    return ParameterIntegrand(dos_fermi_integrand, dos_solver; kws...)
end

dens_params(solver; Σ, β, μ=zero(inv(oneunit(β)))) = (solver, Σ, β, μ)

const DensityFrequencyType = ParameterIntegrand{typeof(dos_fermi_integrand)}

function AutoBZCore.init_solver_cacheval(f::DensityFrequencyType, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end

function (f::DensityFrequencyType)(ω, ::CanonicalParameters)
    canonical_p = (Σ=EtaSelfEnergy(oneunit(ω)), β=inv(oneunit(ω)))
    return ParameterIntegrand(f.f)(ω, canonize(dens_params, merge(canonical_p, f.p)))
end

function AutoBZCore.remake_integrand_cache(f::DensityFrequencyType, dom, p, alg, cacheval, kwargs)
    new_p = canonize(dens_params, p)
    β = new_p[3]
    new_dom = get_safe_fermi_function_limits(β, dom)
    return AutoBZCore.IntegralCache(f, new_dom, new_p, alg, cacheval, kwargs)
end


function dos_fermi_integrand_inside(ω; Σ, h_k, β, μ)
    return dos_fermi_integrand_(ω, dos_integrand(h_k, _evalM(Σ, ω, μ)...), β)
end


struct DensityFrequencyIntegral{T}
    solver::T
end

function (n::DensityFrequencyIntegral)(h_k, dom, Σ, β, μ)
    solver = IntegralSolver(n.solver.f, dom, n.solver.alg, n.solver.cacheval, n.solver.kwargs)
    return solver(; h_k, Σ, β, μ)
end

function ElectronDensityIntegrand(lb, ub, alg, h::AbstractHamiltonianInterp; kwargs...)
    solver_kws, kws = nested_solver_kwargs(NamedTuple(kwargs))
    frequency_integrand = ParameterIntegrand(dos_fermi_integrand_inside; h_k=h(period(h)))
    frequency_solver = IntegralSolver(frequency_integrand, lb, ub, alg; solver_kws...)
    dom = AutoBZCore.PuncturedInterval((lb, ub))
    return FourierIntegrand(DensityFrequencyIntegral(frequency_solver), h, dom; kws...)
end

function ElectronDensityIntegrand(lb, ub, alg, w::FourierWorkspace{<:AbstractHamiltonianInterp}; kwargs...)
    solver_kws, kws = nested_solver_kwargs(NamedTuple(kwargs))
    frequency_integrand = ParameterIntegrand(dos_fermi_integrand_inside; h_k=w(period(w.series)))
    frequency_solver = IntegralSolver(frequency_integrand, lb, ub, alg; solver_kws...)
    dom = AutoBZCore.PuncturedInterval((lb, ub))
    int = DensityFrequencyIntegral(frequency_solver)
    p = ParameterIntegrand(int, dom; kws...)
    nest = make_fourier_nest(p, ParameterIntegrand(int), w)
    return nest === nothing ? FourierIntegrand(p, w) : FourierIntegrand(p, w, nest)
end


const DensityFrequencyInsideType = ParameterIntegrand{typeof(dos_fermi_integrand_inside)}

function AutoBZCore.init_solver_cacheval(f::DensityFrequencyInsideType, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end

function (f::DensityFrequencyInsideType)(ω, ::CanonicalParameters)
    canonical_p = (Σ=EtaSelfEnergy(oneunit(ω)), β=inv(oneunit(ω)), μ=zero(ω))
    return ParameterIntegrand(f.f)(ω, merge(canonical_p, f.p))
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

function dens_params_inside(dom; Σ, β, μ=zero(inv(oneunit(β))))
    new_dom = get_safe_fermi_function_limits(β, dom)
    return (new_dom, Σ, β, μ)
end

const ElectronDensityIntegrandType = FourierIntegrand{<:DensityFrequencyIntegral}

function AutoBZCore.init_solver_cacheval(f::ElectronDensityIntegrandType, dom, alg)
    nest = alloc_nest_buffers(f.nest, interior_point(dom.lims))
    new_f = nest === nothing ? f : FourierIntegrand(f.f, f.w, nest)
    return AutoBZCore.init_cacheval(new_f, dom, CanonicalParameters(), alg)
end

function (f::ElectronDensityIntegrandType)(x::FourierValue, ::CanonicalParameters)
    el = real(_eltype(x.s))
    canonical_p = (Σ=EtaSelfEnergy(oneunit(el)), β=inv(oneunit(el)))
    return FourierIntegrand(f.f.f, f.w)(x, canonize(dens_params_inside, merge(canonical_p, f.f.p)))
end

function AutoBZCore.remake_integrand_cache(f::ElectronDensityIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = canonize(dens_params_inside, p)
    # Define default equispace grid stepping
    new_alg = choose_autoptr_step(alg, sigma_to_eta(p.Σ), parentseries(f.w.series))
    new_cacheval = alg == new_alg ? cacheval : AutoBZCore.init_cacheval(f, dom, p, new_alg)
    return AutoBZCore.IntegralCache(f, dom, new_p, new_alg, new_cacheval, kwargs)
end

function aux_transport_distribution_integrand_(auxfun, vs::SVector{N,V}, Gω₁::G, Gω₂::G) where {N,V,G}
    Aω₁ = spectral_function(Gω₁)
    Aω₂ = spectral_function(Gω₂)
    vsAω₁ = map(v -> v * Aω₁, vs)
    vsAω₂ = map(v -> v * Aω₂, vs)
    return AuxValue(tr_kron(vsAω₁, vsAω₂), auxfun(vs, Gω₁, Gω₂))
end

function default_transport_auxfun(vs, Gω₁, Gω₂)
    vsGω₁ = map(v -> v * Gω₁, vs)
    vsGω₂ = map(v -> v * Gω₂, vs)
    return tr_kron(vsGω₁, vsGω₂)
end

function aux_transport_distribution_integrand_(auxfun, vs::SVector{N,V}, Gω::G) where {N,V,G}
    Aω = spectral_function(Gω)
    vsAω = map(v -> v * Aω, vs)
    return AuxValue(tr_kron(vsAω, vsAω), auxfun(vs, Gω, Gω))
end

function aux_transport_distribution_integrand((h, vs), auxfun, Mω₁, Mω₂, isdistinct)
    if isdistinct
        Gω₁ = gloc_integrand(h, Mω₁)
        Gω₂ = gloc_integrand(h, Mω₂)
        return aux_transport_distribution_integrand_(auxfun, vs, Gω₁, Gω₂)
    else
        Gω = gloc_integrand(h, Mω₁)
        return aux_transport_distribution_integrand_(auxfun, vs, Gω)
    end
end
function aux_transport_distribution_integrand(x::FourierValue, args...)
    return aux_transport_distribution_integrand(x.s, args...)
end

"""
    AuxTransportDistributionIntegrand(hv, [auxfun]; Σ, ω₁, ω₂, μ)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See `Integrand` for more details.
"""
function AuxTransportDistributionIntegrand(hv::AbstractVelocityInterp, auxfun=default_transport_auxfun; kwargs...)
    return FourierIntegrand(aux_transport_distribution_integrand, hv, auxfun; kwargs...)
end
function AuxTransportDistributionIntegrand(w::FourierWorkspace{<:AbstractVelocityInterp}, auxfun=default_transport_auxfun; kwargs...)
    p = ParameterIntegrand(aux_transport_distribution_integrand, auxfun; kwargs...)
    nest = make_fourier_nest(p, ParameterIntegrand(aux_transport_distribution_integrand), w)
    return nest === nothing ? FourierIntegrand(p, w) : FourierIntegrand(p, w, nest)
end

const AuxTransportDistributionIntegrandType = FourierIntegrand{typeof(aux_transport_distribution_integrand)}

function AutoBZCore.init_solver_cacheval(f::AuxTransportDistributionIntegrandType, dom, alg)
    nest = alloc_nest_buffers(f.nest, interior_point(dom.lims))
    new_f = nest === nothing ? f : FourierIntegrand(f.f, f.w, nest)
    return AutoBZCore.init_cacheval(new_f, dom, CanonicalParameters(), alg)
end

auxevalM2(auxfun; kws...) = (auxfun, evalM2(; kws...)...)

function (f::AuxTransportDistributionIntegrandType)(x::FourierValue, ::CanonicalParameters)
    el = real(_eltype(x.s[1]))
    canonical_p = (Σ=EtaSelfEnergy(oneunit(el)), ω₁=zero(el), ω₂=zero(el))
    return FourierIntegrand(f.f.f, f.w)(x, canonize(auxevalM2, merge(canonical_p, f.f.p)))
end

function AutoBZCore.remake_integrand_cache(f::AuxTransportDistributionIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = canonize(auxevalM2, p)
    # Define default equispace grid stepping
    new_alg = choose_autoptr_step(alg, sigma_to_eta(p.Σ), f.w.series)
    new_cacheval = alg == new_alg ? cacheval : AutoBZCore.init_cacheval(f, dom, p, new_alg)
    return AutoBZCore.IntegralCache(f, dom, new_p, new_alg, new_cacheval, kwargs)
end

SymRep(Γ::AuxTransportDistributionIntegrandType) = coord_to_rep(Γ.w.series)


function aux_kinetic_coefficient_integrand(ω, auxfun, Σ, hv_k, n, β, Ω, μ)
    Γ = aux_transport_distribution_integrand(hv_k, auxevalM2(auxfun; Σ, ω₁=ω, ω₂=ω+Ω, μ)...)
    return (ω*β)^n * fermi_window(β, ω, Ω) * Γ
end

"""
    AuxKineticCoefficientIntegrand

A kinetic coefficient integrand that is more robust to the peak-missing problem. See
[`KineticCoefficientIntegrand`](@ref) for arguments.
"""
function AuxKineticCoefficientIntegrand(bz, alg::AutoBZAlgorithm, hv::Union{T,FourierWorkspace{T}}, auxfun=default_transport_auxfun; kwargs...) where {T<:AbstractVelocityInterp}
    solver_kws, kws = nested_solver_kwargs(NamedTuple(kwargs))
    # put the frequency integral outside if the provided algorithm is for the BZ
    transport_integrand = AuxTransportDistributionIntegrand(hv, auxfun)
    transport_solver = IntegralSolver(transport_integrand, bz, alg; solver_kws...)
    return ParameterIntegrand(transport_fermi_integrand, transport_solver; kws...)
end

function aux_transport_fermi_integrand_inside(ω, auxfun; Σ, n, β, Ω, μ, hv_k)
    Γ = aux_transport_distribution_integrand(hv_k, auxevalM2(auxfun; Σ, ω₁=ω, ω₂=ω+Ω, μ)...)
    return transport_fermi_integrand_(ω, Γ, n, β, Ω)
end

function AuxKineticCoefficientIntegrand(lb_, ub_, alg, hv::AbstractVelocityInterp, auxfun=default_transport_auxfun; kwargs...)
    solver_kws, kws = nested_solver_kwargs(NamedTuple(kwargs))
    # put the frequency integral inside otherwise
    frequency_integrand = ParameterIntegrand(aux_transport_fermi_integrand_inside, auxfun; hv_k=hv(period(hv)))
    frequency_solver = IntegralSolver(frequency_integrand, lb_, ub_, alg; solver_kws...)
    dom = AutoBZCore.PuncturedInterval((lb_, ub_))
    return FourierIntegrand(KCFrequencyIntegral(frequency_solver), hv, dom; kws...)
end
function AuxKineticCoefficientIntegrand(lb_, ub_, alg, w::FourierWorkspace{<:AbstractVelocityInterp}, auxfun=default_transport_auxfun; kwargs...)
    solver_kws, kws = nested_solver_kwargs(NamedTuple(kwargs))
    # put the frequency integral inside otherwise
    frequency_integrand = ParameterIntegrand(aux_transport_fermi_integrand_inside, auxfun; hv_k=w(period(w.series)))
    dom = AutoBZCore.PuncturedInterval((lb_, ub_))
    frequency_solver = IntegralSolver(frequency_integrand, dom, alg; solver_kws...)
    int = KCFrequencyIntegral(frequency_solver)
    p = ParameterIntegrand(int, dom; kws...)
    nest = make_fourier_nest(p, ParameterIntegrand(int), w)
    return nest === nothing ? FourierIntegrand(p, w) : FourierIntegrand(p, w, nest)
end

const AuxKCFrequencyInsideType = ParameterIntegrand{typeof(aux_transport_fermi_integrand_inside)}

function AutoBZCore.init_solver_cacheval(f::AuxKCFrequencyInsideType, dom, alg)
    return AutoBZCore.init_cacheval(f, dom, CanonicalParameters(), alg)
end

function (f::AuxKCFrequencyInsideType)(ω, ::CanonicalParameters)
    canonical_p = (Σ=EtaSelfEnergy(oneunit(ω)), n=0, β=inv(oneunit(ω)), Ω=zero(ω), μ=zero(ω))
    return ParameterIntegrand(f.f)(ω, merge(canonical_p, f.p))
end

"""
    AuxOpticalConductivityIntegrand

Returns a `AuxKineticCoefficientIntegrand` with `n=0`. See
[`AuxKineticCoefficientIntegrand`](@ref) for further details
"""
AuxOpticalConductivityIntegrand(args...; kws...) = AuxKineticCoefficientIntegrand(args...; kws..., n=0)

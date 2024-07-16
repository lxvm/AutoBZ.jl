propagator_denominator(h, M) = M-h
propagator_denominator(h::Eigen, M::UniformScaling) =
    propagator_denominator(Diagonal(h.values), M)
propagator_denominator((h, U)::Eigen, M::AbstractMatrix) =
    propagator_denominator(Diagonal(h), U' * M * U) # rotate to Hamiltonian gauge
# propagator_denominator(h::FourierValue, M) = propagator_denominator(h.s, M)

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
See [`trgloc_integrand`](@ref).
"""
dos_integrand(G) = -imag(tr_inv(G))/pi

# shift energy by chemical potential, but not self energy
_evalM(Σ::Union{AbstractMatrix,UniformScaling}, ω, μ) = (ω+μ)*I - Σ
_evalM(Σ::AbstractSelfEnergy, ω, μ) = _evalM(Σ(ω), ω, μ)
evalM(; Σ, ω, μ=zero(ω)) = _evalM(Σ, ω, μ)

function update_gloc!(solver; ω, μ=zero(ω))
    Σ = solver.p[1]
    solver.p = (Σ, evalM(; ω, Σ, μ))
    return
end

# TODO implement these integrands with a CommonSolveIntegralFunction with a ResolventProblem

# Generic behavior for single Green's function integrands (methods and types)
for (name, rep) in (("Gloc", UnknownRep()), ("DiagGloc", UnknownRep()), ("TrGloc", TrivialRep()), ("DOS", TrivialRep()))
    # create and export symbols
    f = Symbol(lowercase(name), "_integrand")
    T = Symbol(name, "Solver")

    @eval begin
        # define a method to evaluate the Green's function
        $f(k, h, (Σ, M)) = $f(propagator_denominator(h, M))
        # We don't want to evalM at every k-point and instead will use
        # AutoBZCore.remake_cache below to evalM once per integral
        # However,these methods are useful for initialization/plotting

        # Define a constructor for the integrand
        """
            $($T)(Σ, h, bz, bzalg; ω, μ=0, kws...)

        Green's function integrands accepting a self energy Σ that can either be a matrix or a
        function of ω (see the self energy section of the documentation for examples)
        Additional keywords are passed directly to the solver.
        Use `AutoBZ.update_gloc!(solver; ω, μ=0)` to change the parameters.
        """
        function $T(Σ::AbstractSelfEnergy, h::AbstractHamiltonianInterp, bz, alg; ω, μ=zero(ω), kws...)
            p = (Σ, evalM(; ω, Σ, μ))
            k = SVector(period(h))
            proto = $f(k, h(k), p)
            f = FourierIntegralFunction($f, h, proto)
            prob = AutoBZProblem($rep, f, bz, p; kws...)
            return init(prob, alg)
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
function choose_autoptr_step(alg::AutoPTR, η::Number, hv::AbstractVelocityInterp)
    return choose_autoptr_step(alg, η, parentseries(hv))
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

function transport_function_integrand((h, vs)::Tuple{Eigen,SVector}; β, μ)
    f′ = Diagonal(β .* fermi′.(β .* (h.values .- μ)))
    f′vs = map(v -> f′*v, vs)
    return tr_kron(vs, f′vs)
end
transport_function_integrand(k, hv, p) = transport_function_integrand(hv; p...)
function update_tf!(solver; β, μ=zero(inv(oneunit(β))))
    solver.p = (; β, μ)
    return
end
"""
    TransportFunctionSolver(hv::AbstractVelocityInterp, bz, bzalg; β, μ=0, kws...)

A function whose integral over the BZ gives the transport function, proportional
to the Drude weight,
```math
D_{\\alpha\\beta} = \\sum_{nm} \\int_{\\text{BZ}} dk f'(\\epsilon_{nk}-\\mu) \\nu_{n\\alpha}(k) \\nu_{m\\beta}(k)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distribution.
Additional keywords are passed directly to the solver.
Use `AutoBZ.update_tf!(solver; β, μ=0)` to update the parameters.
"""
function TransportFunctionSolver(hv::AbstractVelocityInterp, bz, bzalg; β, μ=zero(inv(oneunit(β))), kws...)
    @assert gauge(hv) isa Hamiltonian
    p = (; β, μ)
    k = SVector(period(hv))
    hvk = hv(k)
    proto = transport_function_integrand(k, hvk, p)
    f = FourierIntegralFunction(transport_function_integrand, hv, proto)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, p; kws...)
    return init(prob, bzalg)
end


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
spectral_function(h, M) = spectral_function(gloc_integrand(propagator_denominator(h, M)))

function transport_distribution_integrand(k, v, (Σ, (Mω₁, Mω₂, isdistinct)))
    h, vs = v
    if isdistinct
        Aω₁ = spectral_function(h, Mω₁)
        Aω₂ = spectral_function(h, Mω₂)
        return transport_distribution_integrand_(vs, Aω₁, Aω₂)
    else
        Aω = spectral_function(h, Mω₁)
        return transport_distribution_integrand_(vs, Aω)
    end
end

function _evalM2(Σ, ω₁::T, ω₂::T, μ::T) where {T}
    M = _evalM(Σ, ω₁, μ)
    if ω₁ == ω₂
        (M, M, false)
    else
        (M, _evalM(Σ, ω₂, μ), true)
    end
end
evalM2(; Σ, ω₁, ω₂, μ=zero(ω₁)) = _evalM2(Σ, promote(ω₁, ω₂, μ)...)

function update_td!(solver; ω₁, ω₂, μ=zero(ω₁))
    Σ = solver.p[1]
    solver.p = (Σ, evalM2(; Σ, ω₁, ω₂, μ))
    return
end

"""
    TransportDistributionSolver(Σ, hv::AbstractVelocityInterp, bz, bzalg; ω₁, ω₂, μ=0, kws...)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
Additional keywords are passed directly to the solver.
Use `AutoBZ.update_td!(solver; ω₁, ω₂, μ=0)` to update the parameters.
"""
function TransportDistributionSolver(Σ::AbstractSelfEnergy, hv::AbstractVelocityInterp, bz, bzalg; ω₁, ω₂, μ=zero(ω₁), kwargs...)
    p = (Σ, evalM2(; Σ, ω₁, ω₂, μ))
    k = SVector(period(hv))
    hvk = hv(k)
    proto = transport_distribution_integrand(k, hvk, p)
    f = FourierIntegralFunction(transport_distribution_integrand, hv, proto)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, p; kwargs...)
    return init(prob, bzalg)
end

"""
    KineticCoefficientSolver(hv, bz, bzalg, Σ, falg; n, β, Ω, μ=0, kws...)
    KineticCoefficientSolver(Σ, falg, hv, bz, bzalg; n, β, Ω, μ=0, kws...)

A solver for kinetic coefficients.
The two orderings of arguments correspond to orders of integration.
(The outer integral appears first in the argument list.)
Use `AutoBZ.update_kc!(solver; β, Ω, μ, n)` to change parameters.

Mathematically, this computes
```math
A_{n,\\alpha\\beta}(\\Omega) = \\int_{-\\infty}^{\\infty} d \\omega (\\beta\\omega)^{n} \\frac{f(\\omega) - f(\\omega+\\Omega)}{\\Omega} \\Gamma_{\\alpha\\beta}(\\omega, \\omega+\\Omega)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
"""
function KineticCoefficientSolver(Σ::AbstractSelfEnergy, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm; β, Ω, n, μ=zero(Ω), kws...)
    inner_kws = _rescale_abstol(inv(max(Ω, inv(β))); kws...)
    td_solver = TransportDistributionSolver(Σ, hv, bz, bzalg; ω₁=zero(Ω), ω₂=Ω, μ, inner_kws...)
    p = (; β, μ, Ω, n)
    proto = (zero(Ω)*β)^n * fermi_window(β, zero(Ω), Ω) * td_solver.f.prototype
    f = IntegralFunction(proto) do ω, (; β, μ, Ω, n)
        update_td!(td_solver; ω₁=ω, ω₂=ω+Ω, μ)
        Γ = solve!(td_solver).value
        return (ω*β)^n * fermi_window(β, ω, Ω) * Γ
    end
    prob = IntegralProblem(f, get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ)), p; kws...)
    return init(prob, falg)
end

function update_kc!(solver::AutoBZCore.IntegralSolver; β, Ω, n, μ=zero(Ω))
    Σ = solver.f.f.td_solver.p[1]
    solver.p = (; β, μ, Ω, n)
    solver.dom = get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ))
    return
end

function KineticCoefficientSolver(hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, Σ::AbstractSelfEnergy, falg; β, Ω, n, μ=zero(inv(oneunit(β))), bandwidth=one(μ), kws...)
    k = SVector(period(hv))
    hvk = hv(k)
    proto = (zero(Ω)*β)^n * fermi_window(β, zero(Ω), Ω) * transport_distribution_integrand(k, hvk, (Σ, evalM2(; Σ, ω₁=zero(Ω), ω₂=Ω, μ)))
    f = IntegralFunction(proto) do ω, (Σ, hv, (; β, μ, Ω, n))
        Γ = transport_distribution_integrand(k, hv, (Σ, evalM2(; Σ, ω₁=ω, ω₂=ω+Ω, μ)))
        return (ω*β)^n * fermi_window(β, ω, Ω) * Γ
    end
    V = abs(det(bz.B))
    inner_kws = _rescale_abstol(inv(V*nsyms(bz)); kws...)
    fprob = IntegralProblem(f, get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ)), (Σ, hvk, (; β, μ, Ω, n)); inner_kws...)
    up = (solver, k, hv, p) -> begin
    #     if iszero(Ω) && isinf(β)
    #         # we pass in β=4 since fermi_window(4,0,0)=1, the weight of the delta
    #         # function, and also this prevents (0*β)^n from giving NaN when n!=0
    #         return Ω * f.f(Ω, MixedParameters(; Σ, n, β=4*oneunit(β), Ω, μ, hv_k))
    #     end
        Σ = solver.p[1]
        solver.p = (Σ, hv, p)
        solver.dom = get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ))
        return
    end
    post = (sol, k, h, p) -> sol.value
    f = CommonSolveFourierIntegralFunction(fprob, falg, up, post, hv, proto*Ω)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, (; β, μ, Ω, n); kws...)
    return init(prob, bzalg)
end

function update_kc!(solver::AutoBZCore.AutoBZCache; β, Ω, n, μ=zero(Ω))
    solver.p = (; β, μ, Ω, n)
    return
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

"""
    OpticalConductivitySolver(hv, bz, bzalg, Σ, falg; β, Ω, μ=0, kws...)
    OpticalConductivitySolver(Σ, falg, hv, bz, bzalg; β, Ω, μ=0, kws...)

A solver for the optical conductivity. For details see [`KineticCoefficientSolver`](@ref)
and note that by default the parameter `n=0`. Use `AutoBZ.update_oc!(solver; β, Ω, μ)` to
change parameters.
"""
OpticalConductivitySolver(args...; kws...) = KineticCoefficientSolver(args...; kws..., n=0)
update_oc!(solver; kws...) = update_kc!(solver; kws..., n=0)

# Electron density

"""
    ElectronDensitySolver(h, bz, bzalg, Σ, fdom, falg; β, μ=0, kws...)
    ElectronDensitySolver(Σ, fdom, falg, h, bz, bzalg; β, μ=0, kws...)

A solver for the electron density.
The two orderings of arguments correspond to orders of integration.
(The outer integral appears first in the argument list.)
Use `AutoBZ.update_density!(solver; β, μ=0)`

A function whose integral over the BZ gives the electron density.
Mathematically, this computes
```math
n(\\mu) = \\int_{-\\infty}^{\\infty} d \\omega f(\\omega) \\operatorname{DOS}(\\omega+\\mu)
```
where ``f(\\omega) = (e^{\\beta\\omega}+1)^{-1}`` is the Fermi distriubtion.
To get the density/number of electrons, multiply the result of this integral by `n_sp/det(bz.B)`
"""
function ElectronDensitySolver(Σ::AbstractSelfEnergy, fdom, falg, h::AbstractHamiltonianInterp, bz, bzalg; β, μ=zero(inv(oneunit(β))), bandwidth=one(μ), kws...)
    # TODO better estimate the bandwidth
    V = abs(det(bz.B))
    inner_kws = _rescale_abstol(inv(bandwidth); kws...)
    dos_solver = DOSSolver(Σ, h, bz, bzalg; ω=(fdom[1]+fdom[2])/2, μ, inner_kws...)
    p = (; β, μ)
    proto = dos_solver.f.prototype * V * fermi(β, (fdom[1]+fdom[2])/2)
    f = IntegralFunction(proto) do ω, (; β, μ)
        update_gloc!(dos_solver; ω, μ)
        dos = solve!(dos_solver).value
        return dos*fermi(β, ω)
    end
    prob = IntegralProblem(f, get_safe_fermi_function_limits(β, lb(Σ), ub(Σ)), p; kws...)
    return init(prob, falg)
end

function _rescale_abstol(s; kws...)
    haskey(NamedTuple(kws), :abstol) || return (; kws...)
    return (; kws..., abstol=NamedTuple(kws).abstol*s)
end

function update_density!(solver::AutoBZCore.IntegralSolver; β, μ=zero(inv(oneunit(β))))
    solver.p = (; β, μ)
    solver.dom = get_safe_fermi_function_limits(β, solver.dom...)
    # TODO rescale inner tolerance
    return
end

function ElectronDensitySolver(h::AbstractHamiltonianInterp, bz, bzalg, Σ::AbstractSelfEnergy, fdom, falg; β, μ=zero(inv(oneunit(β))), bandwidth=one(μ), kws...)
    V = abs(det(bz.B))
    k = period(h)
    hk = h(k)
    proto = dos_integrand(propagator_denominator(hk, _evalM(Σ(zero(μ)), zero(μ), μ)))*fermi(β, (fdom[1]+fdom[2])/2)
    f = IntegralFunction(proto) do ω, (Σ, h, (; β, μ))
        return fermi(β, ω)*dos_integrand(propagator_denominator(h, _evalM(Σ(ω), ω, μ)))
    end
    inner_kws = _rescale_abstol(inv(V*nsyms(bz)); kws...)
    fprob = IntegralProblem(f, fdom, (Σ, hk, (; β, μ)); inner_kws...)
    up = (solver, k, h, p) -> begin
        solver.p = (solver.p[1], h, p)
        solver.dom = get_safe_fermi_function_limits(β, fdom...)
        return
    end
    post = (sol, k, h, p) -> sol.value
    f = CommonSolveFourierIntegralFunction(fprob, falg, up, post, h, proto*μ)
    prob = AutoBZProblem(TrivialRep(), f, bz, (; β, μ); kws...)
    return init(prob, bzalg)
end

function update_density!(solver::AutoBZCore.AutoBZCache; β, μ=zero(inv(oneunit(β))))
    solver.p = (; β, μ)
    return
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

function aux_transport_distribution_integrand(k, v, (auxfun, Σ, (Mω₁, Mω₂, isdistinct)))
    h, vs = v
    if isdistinct
        Gω₁ = gloc_integrand(propagator_denominator(h, Mω₁))
        Gω₂ = gloc_integrand(propagator_denominator(h, Mω₂))
        Aω₁ = spectral_function(Gω₁)
        Aω₂ = spectral_function(Gω₂)
        return AutoBZCore.IteratedIntegration.AuxValue(transport_distribution_integrand_(vs, Aω₁, Aω₂), auxfun(vs, Gω₁, Gω₂))
    else
        Gω = gloc_integrand(propagator_denominator(h, Mω₁))
        Aω = spectral_function(Gω)
        return AutoBZCore.IteratedIntegration.AuxValue(transport_distribution_integrand_(vs, Aω), auxfun(vs, Gω, Gω))
    end
end

"""
    AuxTransportDistributionSolver([auxfun], Σ, hv::AbstractVelocityInterp, bz, bzalg; ω₁, ω₂, μ=0, kws...)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
Additional keywords are passed directly to the solver.
Use `AutoBZ.update_auxtd!(solver; ω₁, ω₂, μ)` to update the parameters.
"""
function AuxTransportDistributionSolver(auxfun, Σ::AbstractSelfEnergy, hv::AbstractVelocityInterp, bz, bzalg; ω₁, ω₂, μ=zero(ω₁), kwargs...)
    p = (auxfun, Σ, evalM2(; Σ, ω₁, ω₂, μ))
    k = SVector(period(hv))
    hvk = hv(k)
    proto = aux_transport_distribution_integrand(k, hvk, p)
    f = FourierIntegralFunction(aux_transport_distribution_integrand, hv, proto)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, p; kwargs...)
    return init(prob, bzalg)
end
AuxTransportDistributionSolver(Σ::AbstractSelfEnergy, hv::AbstractVelocityInterp, bz, bzalg; kws...) = AuxTransportDistributionSolver(_trG_auxfun, Σ, hv, bz, bzalg; _trG_kws(; kws...)...)
_trG_auxfun(vs, Gω₁, Gω₂) = tr(Gω₁) + tr(Gω₂)
function _trG_kws(; kws...)
    (!haskey(kws, :abstol) || !(kws[:abstol] isa AutoBZCore.IteratedIntegration.AuxValue)) && @warn "pick a sensible default auxiliary tolerance"
    return (; kws...)
end

function update_auxtd!(solver; ω₁, ω₂, μ=zero(ω₁))
    Σ = solver.p[2]
    solver.p = (solver.p[1], Σ, evalM2(; Σ, ω₁, ω₂, μ))
    return
end

"""
    AuxKineticCoefficientSolver([auxfun], hv, bz, bzalg, Σ, falg; n, β, Ω, μ=0, kws...)
    AuxKineticCoefficientSolver([auxfun], Σ, falg, hv, bz, bzalg; n, β, Ω, μ=0, kws...)

A solver for kinetic coefficients using an auxiliary integrand.
The two orderings of arguments correspond to orders of integration.
(The outer integral appears first in the argument list.)
The default `auxfun` is the sum of the Green's functions.
Use `AutoBZ.update_auxkc!(solver; β, Ω, μ, n)` to change parameters.
"""
function AuxKineticCoefficientSolver(auxfun, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, Σ::AbstractSelfEnergy, falg; β, Ω, n, μ=zero(inv(oneunit(β))), bandwidth=one(μ), kws...)
    k = SVector(period(hv))
    hvk = hv(k)
    proto = (zero(Ω)*β)^n * fermi_window(β, zero(Ω), Ω) * aux_transport_distribution_integrand(k, hvk, (auxfun, Σ, evalM2(; Σ, ω₁=zero(Ω), ω₂=Ω, μ)))
    f = IntegralFunction(proto) do ω, (Σ, hv, (; β, μ, Ω, n))
        Γ = aux_transport_distribution_integrand(k, hv, (auxfun, Σ, evalM2(; Σ, ω₁=ω, ω₂=ω+Ω, μ)))
        return (ω*β)^n * fermi_window(β, ω, Ω) * Γ
    end
    V = abs(det(bz.B))
    inner_kws = _rescale_abstol(inv(V*nsyms(bz)); kws...)
    fprob = IntegralProblem(f, get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ)), (Σ, hvk, (; β, μ, Ω, n)); inner_kws...)
    up = (solver, k, hv, p) -> begin
    #     if iszero(Ω) && isinf(β)
    #         # we pass in β=4 since fermi_window(4,0,0)=1, the weight of the delta
    #         # function, and also this prevents (0*β)^n from giving NaN when n!=0
    #         return Ω * f.f(Ω, MixedParameters(; Σ, n, β=4*oneunit(β), Ω, μ, hv_k))
    #     end
        Σ = solver.p[1]
        solver.p = (Σ, hv, p)
        solver.dom = get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ))
        return
    end
    post = (sol, k, h, p) -> sol.value
    f = CommonSolveFourierIntegralFunction(fprob, falg, up, post, hv, proto*Ω)
    prob = AutoBZProblem(coord_to_rep(coord(hv)), f, bz, (; β, μ, Ω, n); kws...)
    return init(prob, bzalg)
end
function AuxKineticCoefficientSolver(auxfun, Σ::AbstractSelfEnergy, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm; β, Ω, n, μ=zero(Ω), kws...)
    inner_kws = _rescale_abstol(inv(max(Ω, inv(β))); kws...)
    auxtd_solver = AuxTransportDistributionSolver(auxfun, Σ, hv, bz, bzalg; ω₁=zero(Ω), ω₂=Ω, μ, inner_kws...)
    p = (; β, μ, Ω, n)
    proto = (zero(Ω)*β)^n * fermi_window(β, zero(Ω), Ω) * auxtd_solver.f.prototype
    f = IntegralFunction(proto) do ω, (; β, μ, Ω, n)
        update_auxtd!(auxtd_solver; ω₁=ω, ω₂=ω+Ω, μ)
        Γ = solve!(auxtd_solver).value
        return (ω*β)^n * fermi_window(β, ω, Ω) * Γ
    end
    prob = IntegralProblem(f, get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ)), p; kws...)
    return init(prob, falg)
end

AuxKineticCoefficientSolver(hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm, Σ::AbstractSelfEnergy, falg; kws...) = AuxKineticCoefficientSolver(_trG_auxfun, hv, bz, bzalg, Σ, falg; _trG_kws(; kws...)...)
AuxKineticCoefficientSolver(Σ::AbstractSelfEnergy, falg, hv::AbstractVelocityInterp, bz, bzalg::AutoBZAlgorithm; kws...) = AuxKineticCoefficientSolver(_trG_auxfun, Σ, falg, hv, bz, bzalg; _trG_kws(; kws...)...)
update_auxkc!(args...; kws...) = update_kc!(args...; kws...)
function update_auxkc!(solver::AutoBZCore.IntegralSolver; β, Ω, n, μ=zero(Ω))
    Σ = solver.f.f.auxtd_solver.p[2]
    solver.p = (; β, μ, Ω, n)
    solver.dom = get_safe_fermi_window_limits(Ω, β, lb(Σ), ub(Σ))
    return
end

"""
    AuxOpticalConductivitySolver([auxfun], hv, bz, bzalg, Σ, falg; β, Ω, μ=0, kws...)
    AuxOpticalConductivitySolver([auxfun], Σ, falg, hv, bz, bzalg; β, Ω, μ=0, kws...)

A solver for the optical conductivity. For details see [`AuxKineticCoefficientSolver`](@ref)
and note that by default the parameter `n=0`. Use `AutoBZ.update_auxoc!(solver; β, Ω, μ)` to
change parameters.
"""
AuxOpticalConductivitySolver(args...; kws...) = AuxKineticCoefficientSolver(args...; kws..., n=0)
update_auxoc!(solver; kws...) = update_auxkc!(solver; kws..., n=0)

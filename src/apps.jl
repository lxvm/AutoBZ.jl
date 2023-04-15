propagator_denominator(h::AbstractMatrix, M) = M-h
propagator_denominator(h::Eigen, M::UniformScaling) =
    propagator_denominator(Diagonal(h.values), M)
propagator_denominator((h, U)::Eigen, M::AbstractMatrix) =
    propagator_denominator(Diagonal(h), U' * M * U) # rotate to Hamiltonian gauge

"""
    gloc_integrand(h, M)

Returns `inv(M-h)` where `M = ω*I-Σ(ω)`
"""
gloc_integrand(h, M) = inv(propagator_denominator(h, M))

"""
    diaggloc_integrand(h, M)

Returns `diag(inv(M-h))` where `M = ω*I-Σ(ω)`
"""
diaggloc_integrand(h, M) = diag_inv(propagator_denominator(h, M))

"""
    trgloc_integrand(h, M)

Returns `tr(inv(M-h))` where `M = ω*I-Σ(ω)`
"""
trgloc_integrand(h, M) = tr_inv(propagator_denominator(h, M))

"""
    dos_integrand(h, M)

Returns `imag(tr(inv(M-h)))/(-pi)` where `M = ω*I-Σ(ω)`. It is unsafe to use
this in the inner integral for small eta due to the localized tails of the
integrand. The default, safe version also integrates the real part which is less
localized, at the expense of a slight slow-down due to complex arithmetic.
See [`AutoBZ.Jobs.safedos_integrand`](@ref).
"""
dos_integrand(h, M) = imag(tr_inv(propagator_denominator(h, M)))/(-pi)

# TODO write a macro to automate the evaluation/canonization of arguments

# shift energy by chemical potential, but not self energy
evalM(Σ::Union{AbstractMatrix,UniformScaling}, ω, μ) = (ω+μ)*I - Σ
evalM(Σ::AbstractSelfEnergy, ω, μ) = evalM(Σ(ω), ω, μ)
evalM(Σ, ω; μ=0) = evalM(Σ, ω, μ)
evalM(Σ; ω, μ=0) = evalM(Σ, ω, μ)

const EvalSelfEnergyType = Union{AbstractSelfEnergy,AbstractMatrix,UniformScaling}

const EvalMType = Union{
    MixedParameters{<:Tuple{EvalSelfEnergyType,Real,Real},NamedTuple{(),Tuple{}}},
    MixedParameters{<:Tuple{EvalSelfEnergyType,Real},NamedTuple{(),Tuple{}}},
    MixedParameters{<:Tuple{EvalSelfEnergyType,Real},<:NamedTuple{(:μ,),<:Tuple{Real}}},
    MixedParameters{<:Tuple{EvalSelfEnergyType},<:NamedTuple{(:ω,),<:Tuple{Real}}},
    MixedParameters{<:Tuple{EvalSelfEnergyType},<:NamedTuple{<:Any,<:Tuple{Real,Real}}},
}

evalM(p::EvalMType) =
    MixedParameters(evalM(getfield(p, :args)...; getfield(p, :kwargs)...))

# Generic behavior for single Green's function integrands (methods and types)
for name in ("Gloc", "DiagGloc", "TrGloc", "DOS")
    # create and export symbols
    f = Symbol(lowercase(name), "_integrand")
    T = Symbol(name, "Integrand")

    # Define the type alias to have the same behavior as the function
    """
    $(T)(h, Σ, ω, μ)
    $(T)(h, Σ, ω; μ=0)
    $(T)(h, Σ; ω, μ=0)

    See `FourierIntegrand` for more details.
    """
    @eval $T(h::HamiltonianInterp, Σ, args...; kwargs...) =
        FourierIntegrand($f, h, Σ, args...; kwargs...)

    # pre-evaluate the self energy when constructing the integrand
    @eval function FourierIntegrand(f::typeof($f), h::HamiltonianInterp, p::EvalMType)
        FourierIntegrand(f, h, evalM(p))
    end

    # Define default equispace grid stepping based
    @eval function npt_update(f::FourierIntegrand{typeof($f)}, npt::Integer)
        η = im_sigma_to_eta(-imag(f.p[1]))
        eta_npt_update(npt, η, period(f.s)[1])
    end
end

# helper functions for equispace updates

im_sigma_to_eta(x::UniformScaling) = -x.λ
im_sigma_to_eta(x::AbstractMatrix) = minimum(abs, x)

"""
    eta_npt_update(npt, η, [n₀=6.0, Δn=2.3])

Implements the heuristics for incrementing kpts suggested in Appendix A
http://arxiv.org/abs/2211.12959. Choice of `n₀≈2π`, close to the period of a
canonical BZ, approximately places a point in every box of size `η`. Choice of
`Δn≈log(10)` should get an extra digit of accuracy from PTR upon refinement.
"""
eta_npt_update(npt, η, n₀=6.0, Δn=2.3) = npt + eta_npt_update_(η, npt == 0 ? n₀ : Δn)
eta_npt_update_(η, c) = max(50, round(Int, c/η))

# transport and conductivity integrands

function transport_function_integrand((h, vs)::Tuple{Eigen,SVector{N,T}}, β, μ) where {N,T}
    f′ = Diagonal(β .* fermi′.(β .* (h.values .- μ)))
    f′vs = map(v -> f′*v, vs)
    tr_kron(f′, f′vs)
end
transport_function_integrand(hvs, β; μ=0) = transport_function_integrand(hvs, β, μ)
transport_function_integrand(hvs; β, μ=0) = transport_function_integrand(hvs, β, μ)

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
    FourierIntegrand(transport_function_integrand, hv, args...; kwargs...)
end

const TransportFunctionIntegrandType = FourierIntegrand{typeof(transport_function_integrand)}

SymRep(D::TransportFunctionIntegrandType) = coord_to_rep(D.s)


function transport_distribution_integrand_(vs::SVector{N,V}, Aω₁::A, Aω₂::A) where {N,V,A}
    vsAω₁ = map(v -> v * Aω₁, vs)
    vsAω₂ = map(v -> v * Aω₂, vs)
    return tr_kron(vsAω₁, vsAω₂)
end

spectral_function(G::AbstractMatrix) = (G - G')/(-2pi*im)   # skew-Hermitian part
spectral_function(G::Diagonal) = imag(G)/(-pi)              # optimization
spectral_function(h, M) = spectral_function(gloc_integrand(h, M))

transport_distribution_integrand((h, vs), Mω₁, Mω₂) =
    transport_distribution_integrand_(vs, spectral_function(h, Mω₁), spectral_function(h, Mω₂))

"""
    TransportDistributionIntegrand(hv, Σ, ω₁, ω₂, μ)
    TransportDistributionIntegrand(hv, Σ, ω₁, ω₂; μ)
    TransportDistributionIntegrand(hv, Σ; ω₁, ω₂, μ)

A function whose integral over the BZ gives the transport distribution
```math
\\Gamma_{\\alpha\\beta}(\\omega_1, \\omega_2) = \\int_{\\text{BZ}} dk \\operatorname{Tr}[\\nu_\\alpha(k) A(k,\\omega_1) \\nu_\\beta(k) A(k, \\omega_2)]
```
Based on [TRIQS](https://triqs.github.io/dft_tools/latest/guide/transport.html).
See `FourierIntegrand` for more details.
"""
TransportDistributionIntegrand(hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    FourierIntegrand(transport_distribution_integrand, hv, Σ, args...; kwargs...)

function evalM2(Σ, ω₁, ω₂, μ)
    M = evalM(Σ, ω₁, μ)
    if ω₁ == ω₂
        (M, M)
    else
        (M, evalM(Σ, ω₂, μ))
    end
end
evalM2(Σ, ω₁, ω₂; μ=0) = evalM2(Σ, ω₁, ω₂, μ)
evalM2(Σ; ω₁, ω₂, μ=0) = evalM2(Σ, ω₁, ω₂, μ)

const EvalM2Type = Union{
    MixedParameters{<:Tuple{EvalSelfEnergyType,Real,Real,Real},NamedTuple{(),Tuple{}}},
    MixedParameters{<:Tuple{EvalSelfEnergyType,Real,Real},NamedTuple{(),Tuple{}}},
    MixedParameters{<:Tuple{EvalSelfEnergyType,Real,Real},<:NamedTuple{(:μ,),<:Tuple{Real}}},
    MixedParameters{<:Tuple{EvalSelfEnergyType},<:NamedTuple{(:ω₁, :ω₂),<:Tuple{Real,Real}}},
    MixedParameters{<:Tuple{EvalSelfEnergyType},<:NamedTuple{(:ω₂, :ω₁),<:Tuple{Real,Real}}},
    MixedParameters{<:Tuple{EvalSelfEnergyType},<:NamedTuple{<:Any,<:Tuple{Real,Real,Real}}},
}

evalM2(p::EvalM2Type) =
    MixedParameters(evalM2(getfield(p, :args)...; getfield(p, :kwargs)...), NamedTuple())
function FourierIntegrand(f::typeof(transport_distribution_integrand), hv::AbstractVelocityInterp, p::EvalM2Type)
    FourierIntegrand(f, hv, evalM2(p))
end

const TransportDistributionIntegrandType = FourierIntegrand{typeof(transport_distribution_integrand)}

SymRep(Γ::TransportDistributionIntegrandType) = coord_to_rep(Γ.s)

function npt_update(Γ::TransportDistributionIntegrandType, npt::Integer)
    ηω₁ = im_sigma_to_eta(-imag(Γ.p[1]))
    ηω₂ = im_sigma_to_eta(-imag(Γ.p[2]))
    eta_npt_update(npt, min(ηω₁, ηω₂), period(Γ.s)[1])
end



transport_fermi_integrand(ω, Γ, n::Real, β::Real, Ω::Real, μ::Real) =
    (ω*β)^n * fermi_window(β, ω, Ω) * Γ(ω, ω+Ω, μ)

kinetic_coefficient_integrand(ω, Σ, hv_k, n::Real, β::Real, Ω::Real, μ::Real) =
    (ω*β)^n * fermi_window(β, ω, Ω) * transport_distribution_integrand(hv_k, evalM2(Σ, ω, ω+Ω, μ)...)

kinetic_coefficient_frequency_integral(hv_k, frequency_solver, n::Real, β::Real, Ω::Real, μ::Real) =
    frequency_solver(hv_k, n, β, Ω, μ)

#=
infnorm(x::AbstractArray) = maximum(norm, x)
function kinetic_coefficient_integrand(ω, Σ, (h, vs), n::Real, β::Real, Ω::Real, μ::Real)
    s = -(ω*β)^n * β * fermi_window(β*ω, β*Ω)/(2pi)^2
    Gω₁ = gloc_integrand(h, evalM(Σ, ω, μ))
    Gω₂ = gloc_integrand(h, evalM(Σ, ω+Ω, μ))
    vsGω₁ = map(v -> v*Gω₁, vs)
    vsGω₂ = map(v -> v*Gω₂, vs)
    vsGω₁′ = map(v -> v*(Gω₁'), vs)
    vsGω₂′ = map(v -> v*(Gω₂'), vs)

    SVector{5}(
        # 10000sqrt.(abs.(s * tr_kron(vsGω₁, vsGω₂))), # auxiliary integrand
        s * tr_kron(vsGω₁, vsGω₂),
        s * tr_kron(vsGω₁, -vsGω₂′),
        s * tr_kron(-vsGω₁′, vsGω₂),
        s * tr_kron(vsGω₁′, vsGω₂′),
        )
    end

kinetic_coefficient_frequency_integral(hv_k, frequency_solver, n::Real, β::Real, Ω::Real, μ::Real) =
    sum(frequency_solver(hv_k, n, β, Ω, μ)[2:5])
=#

"""
    KineticCoefficientIntegrand([bz=FullBZ,] alg::AbstractAutoBZAlgorithm, hv::AbstracVelocity, Σ; n, β, Ω, abstol, reltol, maxiters)
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
function KineticCoefficientIntegrand(bz, alg::AbstractAutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    # put the frequency integral outside if the provided algorithm is for the BZ
    transport_integrand = TransportDistributionIntegrand(hv, Σ)
    transport_solver = IntegralSolver(transport_integrand, bz, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    transport_solver(max(-10.0, lb(Σ)), min(10.0, ub(Σ)), 0) # precompile the solver
    Integrand(transport_fermi_integrand, transport_solver, args...; kwargs...)
end
KineticCoefficientIntegrand(alg::AbstractAutoBZAlgorithm, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    KineticCoefficientIntegrand(FullBZ(2pi*I(ndims(hv))), alg, hv, Σ, args...; kwargs...)

function KineticCoefficientIntegrand(lb_, ub_, alg, hv::AbstractVelocityInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    # put the frequency integral inside otherwise
    # TODO fix warnings so that safe limits of integration are computed only once
    frequency_integrand = Integrand(kinetic_coefficient_integrand, Σ)
    frequency_solver = IntegralSolver(frequency_integrand, lb_, ub_, alg; do_inf_transformation=Val(false), abstol=abstol, reltol=reltol, maxiters=maxiters)
    frequency_solver(hv(fill(0.0, ndims(hv))), 0, Inf, min(10.0, ub(Σ)-lb(Σ)), 0) # precompile the solver
    FourierIntegrand(kinetic_coefficient_frequency_integral, hv, frequency_solver, args...; kwargs...)
end
KineticCoefficientIntegrand(alg, hv::AbstractVelocityInterp, Σ, args...; kwargs...) =
    KineticCoefficientIntegrand(lb(Σ), ub(Σ), alg, hv, Σ, args...; kwargs...)

canonize_kc_params(solver::IntegralSolver, n_, β_, Ω_, μ_; n=n_, β=β_, Ω=Ω_, μ=μ_) = (solver, n, β, Ω, μ)
#=
function canonize_kc_params(solver_::IntegralSolver{iip,}, n_, β_, Ω_, μ_; n=n_, β=β_, Ω=Ω_, μ=μ_)
    iszero(Ω) && isinf(β) && throw(ArgumentError("Ω=0, T=0 not yet implemented. As a workaround, evaluate the KCIntegrand at ω=0"))
    a, b = get_safe_fermi_window_limits(Ω, β, solver_.lb, solver_.ub)
    solver = IntegralSolver(solver_.f, a, b, solver_.alg, sensealg = solver_.sensealg,
            do_inf_transformation = solver_.do_inf_transformation, kwargs = solver_.kwargs,
            abstol = solver_.abstol, reltol = solver_.reltol, maxiters = solver_.maxiters)
    (solver, n, β, Ω, μ)
end
=#
canonize_kc_params(solver::IntegralSolver, n, β, Ω; μ=0) = canonize_kc_params(solver, n, β, Ω, μ)
canonize_kc_params(solver::IntegralSolver, n, β; Ω, μ=0) = canonize_kc_params(solver, n, β, Ω, μ)
canonize_kc_params(solver::IntegralSolver, n; β, Ω, μ=0) = canonize_kc_params(solver, n, β, Ω, μ)
canonize_kc_params(solver::IntegralSolver; n, β, Ω, μ=0) = canonize_kc_params(solver, n, β, Ω, μ)

const CanonizeKCType = Union{
    MixedParameters{<:Tuple{IntegralSolver,Real,Real,Real,Real},<:NamedTuple{<:Any,<:Tuple{Real,Vararg{Real}}}},
    MixedParameters{<:Tuple{IntegralSolver,Real,Real,Real},NamedTuple{(),Tuple{}}},
    MixedParameters{<:Tuple{IntegralSolver,Real,Real,Real},<:NamedTuple{(:μ,),<:Tuple{Real}}},
    MixedParameters{<:Tuple{IntegralSolver,Real,Real},<:NamedTuple{<:Any,<:Tuple{Real,Vararg{Real}}}},
    MixedParameters{<:Tuple{IntegralSolver,Real},<:NamedTuple{<:Any,<:Tuple{Real,Real,Vararg{Real}}}},
    MixedParameters{<:Tuple{IntegralSolver},<:NamedTuple{<:Any,<:Tuple{Real,Real,Real,Vararg{Real}}}},
}

canonize_kc_params(p::CanonizeKCType) =
    MixedParameters(canonize_kc_params(getfield(p, :args)...; getfield(p, :kwargs)...), NamedTuple())
# TODO, if T=Ω=0, intercept this stage and replace with distributional integrand
Integrand(f::typeof(transport_fermi_integrand), p::CanonizeKCType) =
    Integrand(f, canonize_kc_params(p))
FourierIntegrand(f::typeof(kinetic_coefficient_frequency_integral), hv::AbstractVelocityInterp, p::CanonizeKCType) =
    FourierIntegrand(f, hv, canonize_kc_params(p))


const KineticCoefficientIntegrandType = FourierIntegrand{typeof(kinetic_coefficient_frequency_integral)}

SymRep(kc::KineticCoefficientIntegrandType) = coord_to_rep(kc.s)


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
        l = lb
    end
    if u+Ω > ub
        @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from above"
        u = ub-Ω
    end
    l, u
end

# return (Ω, β)
function get_window_params(p::CanonizeKCType, T=Float64)
    pp = canonize_kc_params(p)
    return (convert(T, pp[4]), convert(T, pp[3]))
end
function get_window_params(p::MixedParameters{<:Tuple{EvalSelfEnergyType,Tuple,Real,Real,Real,Real},NamedTuple{(),Tuple{}}}, T=Float64)
    return (convert(T, p[5]), convert(T, p[4]))
end

# provide safe limits of integration for frequency integrals
function construct_problem(s::Union{IntegralSolver{iip,<:Integrand{typeof(transport_fermi_integrand)}},IntegralSolver{iip,<:Integrand{typeof(kinetic_coefficient_integrand)}}}, p::MixedParameters) where iip
    Ω, β = get_window_params(merge(s.f.p, p))
    iszero(Ω) && isinf(β) && throw(ArgumentError("Ω=0, T=0 not yet implemented. As a workaround, evaluate the KCIntegrand at ω=0"))
    a, b = get_safe_fermi_window_limits(Ω, β, s.lb, s.ub)
    IntegralProblem{iip}(s.f, a, b, p; s.kwargs...)
end

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
    fermi(β*ω)*dos(ω, μ)

electron_density_integrand(ω, Σ, h_k, β, μ) =
    fermi(β*ω)*dos_integrand(h_k, evalM(Σ, ω, μ))

electron_density_frequency_integral(h_k, frequency_solver, β, μ) =
    frequency_solver(h_k, β, μ)

"""
    ElectronDensityIntegrand([bz=FullBZ], alg::AbstractAutoBZAlgorithm, h::HamiltonianInterp, Σ; β, [μ=0])
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
function ElectronDensityIntegrand(bz, alg::AbstractAutoBZAlgorithm, h::HamiltonianInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    dos_int = DOSIntegrand(h, Σ)
    dos_solver = IntegralSolver(dos_int, bz, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    dos_solver(0.0, 0) # precompile the solver
    Integrand(dos_fermi_integrand, dos_solver, args...; kwargs...)
end
ElectronDensityIntegrand(alg::AbstractAutoBZAlgorithm, h::HamiltonianInterp, Σ, args...; kwargs...) =
    ElectronDensityIntegrand(FullBZ(2pi*I(ndims(h))), alg, h, Σ, args...; kwargs...)

function ElectronDensityIntegrand(lb, ub, alg, h::HamiltonianInterp, Σ, args...;
    abstol=0.0, reltol=iszero(abstol) ? sqrt(eps()) : zero(abstol), maxiters=typemax(Int), kwargs...)
    frequency_integrand = Integrand(electron_density_integrand, Σ)
    frequency_solver = IntegralSolver(frequency_integrand, lb, ub, alg; abstol=abstol, reltol=reltol, maxiters=maxiters)
    frequency_solver(h(fill(0.0, ndims(h))), 1.0, 0) # precompile the solver
    FourierIntegrand(electron_density_frequency_integral, h, frequency_solver, args...; kwargs...)
end
ElectronDensityIntegrand(alg, h::HamiltonianInterp, Σ, args...; kwargs...) =
    ElectronDensityIntegrand(lb(Σ), ub(Σ), alg, h, Σ, args...; kwargs...)

canonize_density_params(solver::IntegralSolver, β_, μ_; β=β_, μ=μ_) = (solver, β, μ)
canonize_density_params(solver::IntegralSolver, β; μ=0) = canonize_density_params(solver, β, μ)
canonize_density_params(solver::IntegralSolver; β, μ=0) = canonize_density_params(solver, β, μ)

const CanonizeDensityType = Union{
    MixedParameters{<:Tuple{IntegralSolver,Real,Real},<:NamedTuple{<:Any,<:Tuple{Real,Vararg{Real}}}},
    MixedParameters{<:Tuple{IntegralSolver,Real},NamedTuple{(),Tuple{}}},
    MixedParameters{<:Tuple{IntegralSolver,Real},<:NamedTuple{(:μ,),<:Tuple{Real}}},
    MixedParameters{<:Tuple{IntegralSolver},<:NamedTuple{(:β,),<:Tuple{Real}}},
    MixedParameters{<:Tuple{IntegralSolver},<:NamedTuple{<:Any,<:Tuple{Real,Real}}},
}

canonize_density_params(p::CanonizeDensityType) =
    MixedParameters(canonize_density_params(getfield(p, :args)...; getfield(p, :kwargs)...), NamedTuple())
Integrand(f::typeof(dos_fermi_integrand), p::CanonizeDensityType) =
    Integrand(f, canonize_density_params(p))
FourierIntegrand(f::typeof(electron_density_frequency_integral), hv::HamiltonianInterp, p::CanonizeDensityType) =
    FourierIntegrand(f, hv, canonize_density_params(p))

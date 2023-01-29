"""
    AutoBZ.Jobs

A module that implements applications of AutoBZ
"""
module Jobs

using LinearAlgebra
using Printf

using HDF5
using StaticArrays
using QuadGK: quadgk, alloc_segbuf
using Polyhedra: vrep, Line, Ray

using ..AutoBZ

export read_h5_to_nt, write_nt_to_h5
export adaptive_fourier_integration, automatic_equispace_fourier_integration, equispace_fourier_integration, auto_fourier_integration
export run_dos_adaptive, run_dos_auto_equispace, run_dos_equispace, run_dos
export run_kinetic_adaptive, run_kinetic_auto_equispace, run_kinetic_equispace, run_kinetic

include("AdaptChebInterp.jl")
include("EquiBaryInterp.jl")

using .EquiBaryInterp: LocalEquiBaryInterp

include("linalg.jl")
include("fourier3d.jl")
include("band_velocities.jl")
include("self_energies.jl")
include("self_energies_io.jl")
include("wannier90io.jl")
include("fermi.jl")
include("apps.jl")

#=
Section: loading data from HDF5
- NamedTuples
- Hamiltonians
- Self Energies
=#

"""
    read_h5_to_nt(filename)

Loads the h5 archive from `filename` and reads its datasets into a `NamedTuple`
and its groups into `NamedTuple`s recursively.
"""
read_h5_to_nt(filename) = h5open(read_h5_to_nt_, filename, "r")
read_h5_to_nt_(h5) = NamedTuple([Pair(Symbol(key), ((val = h5[key]) isa HDF5.Group) ? read_h5_to_nt_(val) : h5_dset_to_vec(read(h5, key))) for key in keys(h5)])
h5_dset_to_vec(x::Vector) = identity(x)
function h5_dset_to_vec(A::Array{T,N}) where {T,N}
    S = size(A)[1:N-1]
    reinterpret(SArray{Tuple{S...},T,N-1,prod(S)}, vec(A))
end

#=
Section: saving data to HDF5
- NamedTuple to H5
=#

"""
    write_nt_to_h5(nt::NamedTuple, filename)

Takes a `NamedTuple` and writes its values, which must be arrays, into an h5
archive at `filename` with dataset names corresponding to the tuple names.
If a value is a `NamedTuple`, its datasets are written to h5 groups recursively.
"""
write_nt_to_h5(nt::NamedTuple, filename) = h5open(filename, "w") do h5
    write_nt_to_h5_(nt, h5)
end
function write_nt_to_h5_(nt::NamedTuple, h5)
    for key in keys(nt)
        if (val = nt[key]) isa NamedTuple
            write_nt_to_h5_(val, create_group(h5, string(key)))
        else
            write(h5, string(key), vec_to_h5_dset(val))
        end
    end
end
vec_to_h5_dset(x::Number) = vec(collect(x))
vec_to_h5_dset(x::AbstractVector) = collect(x)
vec_to_h5_dset(x::Vector) = identity(x)
vec_to_h5_dset(x::Vector{T}) where {T<:StaticArray} = reshape(reinterpret(eltype(T), x), size(T)..., :)

#=
Section: parallelization
- Batching
=#

"""
    batch_smooth_param(xs, nthreads)

If the cost of a calculation smoothly varies with the parameters `xs`, then
batch `xs` into `nthreads` groups where the `i`th element of group `j` is
`xs[j+(i-1)*nthreads]`
"""
function batch_smooth_param(xs, nthreads)
    batches = [Tuple{Int,eltype(xs)}[] for _ in 1:min(nthreads, length(xs))]
    for (i, x) in enumerate(xs)
        push!(batches[mod(i-1, nthreads)+1], (i, x))
    end
    batches
end

#=
Section: User-defined integral calculations
=#

"""
    adaptive_fourier_integration(integrand, BZ, fs, ps, rtol, atol, [nthreads=Threads.nthreads()])

Returns a `NamedTuple` with names `I, E, t, p` containing the integrals, errors,
and timings for a user-defined `integrand` of a Fourier series `fs` and
parameters `ps`. The integration tolerances are `rtol, atol`. This function
constructs a [`AutoBZ.FourierIntegrand`](@ref) for each parameter value and
calls [`AutoBZ.iterated_integration`](@ref) with limits of integration `BZ` to
get the results.
"""
function adaptive_fourier_integration(integrand, BZ::AbstractBZ{d}, fs, ps, rtol, atol, nthreads=Threads.nthreads(), atols=fill(atol, length(ps)); order=7, initdivs=ntuple(i -> Val(1), Val{d}())) where d
    test = FourierIntegrand(integrand, fs, ps[1])
    T = AutoBZ.iterated_integral_type(test, BZ)
    ints = Vector{T}(undef, length(ps))
    errs = Vector{Float64}(undef, length(ps))
    ts = Vector{Float64}(undef, length(ps))
    @info "Beginning parameter sweep using IAI"
    @info "using $nthreads threads for frequency parallelization"
    batches = batch_smooth_param(ps, nthreads)
    t = time()
    Threads.@threads for batch in batches
        fs_ = deepcopy(fs) # to avoid data races for AbstractFourierSeries3D
        segbufs = AutoBZ.alloc_segbufs(test, BZ)
        for (i, p) in batch
            @info @sprintf "starting parameter %i" i
            t_ = time()
            f = FourierIntegrand(integrand, fs_, p)
            ints[i], errs[i] = iterated_integration(f, BZ; atol=atols[i], rtol=rtol, order=order, initdivs=initdivs, segbufs=segbufs)
            ts[i] = time() - t_
            @info @sprintf "finished parameter %i in %e (s) wall clock time" i ts[i]
        end
    end
    @info @sprintf "Finished parameter sweep in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (I=ints, E=errs, t=ts, p=ps)
end

"""
    automatic_equispace_fourier_integration(integrand, BZ, fs, ps, rtol, atol, [nthreads=1])

Returns a `NamedTuple` with names `I, E, t, p, npt1, npt2` containing the
integrals, errors, timings, and number of kpts per dimension for a user-defined
`integrand` of a Fourier series `fs` and parameters `ps`. The integration
tolerances are `rtol, atol`. This function constructs a
[`AutoBZ.FourierIntegrand`](@ref) for each parameter value and calls
[`AutoBZ.automatic_equispace_integration`](@ref) with limits of integration
`BZ` to get the results.
"""
function automatic_equispace_fourier_integration(integrand, BZ, fs, ps, rtol, atol, nthreads=1)
    test = FourierIntegrand(integrand, fs, ps[1])
    T = AutoBZ.iterated_integral_type(test, BZ)
    ints = Vector{T}(undef, length(ps))
    errs = Vector{Float64}(undef, length(ps))
    ts = Vector{Float64}(undef, length(ps))
    npt1s = Vector{Int}(undef, length(ps))
    npt2s = Vector{Int}(undef, length(ps))
    
    @info "Beginning parameter sweep using automatic PTR"
    @info "using $nthreads threads for frequency parallelization"
    t = time()
    batches = batch_smooth_param(ps, nthreads)
    Threads.@threads for batch in batches
        fs_ = deepcopy(fs) # to avoid data races for AbstractFourierSeries3D
        # unroll first iteration to get the pre-evaluation buffer
        @info @sprintf "starting parameter %i" batch[1][1]
        t_ = time()
        f = FourierIntegrand(integrand, fs_, batch[1][2])
        ints[1], errs[1], rule_buf = automatic_equispace_integration(f, BZ; atol=atol, rtol=rtol)
        ts[1] = time() - t_
        npt1s[1] = rule_buf.npt1
        npt2s[1] = rule_buf.npt2
        for (i, p) in batch
            @info @sprintf "starting parameter %i" i
            t_ = time()
            f = FourierIntegrand(integrand, fs_, p)
            ints[i], errs[i], rule_buf = automatic_equispace_integration(f, BZ; atol=atol, rtol=rtol, rule_buf...)
            ts[i] = time() - t_
            npt1s[i] = rule_buf.npt1
            npt2s[i] = rule_buf.npt2
            @info @sprintf "finished parameter %i in %e (s) wall clock time" i ts[i]
        end
    end
    @info @sprintf "Finished parameter sweep in %e (s) wall clock time" (time()-t)
    (I=ints, E=errs, t=ts, npt1=npt1s, npt2=npt2s, p=ps)
end

"""
    equispace_fourier_integration(integrand, BZ, fs, ps, npt, [nthreads=Threads.nthreads()])

Returns a `NamedTuple` with names `I, t, p` containing the integrals and timings
for a user-defined `integrand` of a Fourier series `fs` and parameters `ps`.
`npt` is the number of integration nodes per dimension. This function constructs
a [`AutoBZ.FourierIntegrand`](@ref) for each parameter value and calls
[`AutoBZ.equispace_integration`](@ref) with limits of integration `BZ` to get
the results. The caller should check that the result is converged with respect
to `npt`.
"""
function equispace_fourier_integration(integrand, BZ, fs, ps, npt, nthreads=Threads.nthreads())
    test = FourierIntegrand(integrand, fs, ps[1])
    T = AutoBZ.iterated_integral_type(test, BZ)
    ints = Vector{T}(undef, length(ps))
    ts = Vector{Float64}(undef, length(ps))
    
    @info "Beginning parameter sweep using PTR with $npt kpts per dim"
    
    @info "pre-evaluating Fourier series..."
    t = time()
    rule = equispace_rule(test, BZ, npt)
    @info "finished pre-evaluating Fourier series in $(time() - t) (s)"
    
    @info "using $nthreads threads for frequency parallelization"
    t = time()
    batches = batch_smooth_param(ps, nthreads)
    Threads.@threads for batch in batches
        for (i, p) in batch
            @info @sprintf "starting parameter %i" i
            t_ = time()
            f = FourierIntegrand(integrand, fs, p)
            ints[i], = equispace_integration(f, BZ; npt=npt, rule=rule)
            ts[i] = time() - t_
            @info @sprintf "finished parameter %i in %e (s) wall clock time" i ts[i]
        end
    end
    @info @sprintf "Finished parameter sweep in %e (s) wall clock time" (time()-t)
    (I=ints, t=ts, p=ps)
end


"""
    auto_fourier_integration(integrand, BZ, fs, ps, rtol, atol, [nthreads=Threads.nthreads()]; ertol=1.0, eatol=0.0)

Returns a `NamedTuple` with names `I, E, t, p` containing the integrals, errors,
and timings for obtained for a user-defined `integrand` of a Fourier series `fs`
and parameters `ps` (see [`AutoBZ.WannierIntegrand`](@ref)). The integration
tolerances are `atol, rtol`. The integral is evaluated adaptively. However if
`rtol` is nonzero, it is first estimated by an equispace method using wide
tolerances `eatol` and `ertol`, and then uses a narrow absolute tolerance set by
`max(atol,rtol*norm(int))` to evaluate and return the adaptive integral.
"""
function auto_fourier_integration(integrand, BZ::AbstractBZ{d}, fs, ps, rtol, atol, nthreads=Threads.nthreads(); ertol=1.0, eatol=0.0, order=7, initdivs=ntuple(i -> Val(1), Val{d}())) where d
    rtol == 0 && return adaptive_fourier_integration(integrand, BZ, fs, ps, rtol, atol, nthreads; order=order, initdivs=initdivs)
    equi_results = automatic_equispace_fourier_integration(integrand, BZ, fs, ps, ertol, eatol)
    results = adaptive_fourier_integration(integrand, BZ, fs, ps, 0.0, 0.0, nthreads, [max(atol, rtol*norm(I)) for I in equi_results.I]; order=order, initdivs=initdivs)
    (I=results.I, E=results.E, t=results.t, pre_I=equi_results.I, pre_E=equi_results.E, pre_t=equi_results.t, npt1=equi_results.npt1, npt2=equi_results.npt2, p=ps)
end


#=
Section: DOS calculations
- Convergence of PTR at various omega
=#

"""
    run_dos_adaptive(H, Σ::AbstractSelfEnergy, ωs, BZ, rtol, atol, [nthreads=Threads.nthreads()])

Returns a `NamedTuple` with names `I, E, t, omega` containing the results,
errors, and timings for a density of states calculation done at frequencies `ωs`
with parameters `atol, rtol`. This function constructs a [`AutoBZ.DOSIntegrand`](@ref) for
each parameter value and calls [`AutoBZ.iterated_integration`](@ref) on it over the domain of
the `BZ` to get the results.
"""
function run_dos_adaptive(H, Σ::AbstractSelfEnergy, ωs, BZ::AbstractBZ{d}, rtol, atol, nthreads=Threads.nthreads(), atols=fill(atol, length(ωs)); order=7, initdivs=ntuple(i -> Val(1), Val{d}())) where d
    test = DOSIntegrand(H, Σ, zero(eltype(ωs)))
    T = AutoBZ.iterated_integral_type(test, BZ)
    ints = Vector{T}(undef, length(ωs))
    errs = Vector{Float64}(undef, length(ωs))
    ts = Vector{Float64}(undef, length(ωs))
    @info "Beginning DOS frequency sweep using IAI"
    @info "using $nthreads threads for frequency parallelization"
    batches = batch_smooth_param(ωs, nthreads)
    t = time()
    Threads.@threads for batch in batches
        H_ = deepcopy(H) # to avoid data races for AbstractFourierSeries3D
        segbufs = AutoBZ.alloc_segbufs(test, BZ)
        for (i, ω) in batch
            @info @sprintf "starting ω=%e" ω
            t_ = time()
            D = DOSIntegrand(H_, Σ, ω)
            ints[i], errs[i] = iterated_integration(D, BZ; atol=atols[i], rtol=rtol, order=order, initdivs=initdivs, segbufs=segbufs)
            ts[i] = time() - t_
            @info @sprintf "finished ω=%e in %e (s) wall clock time" ω ts[i]
        end
    end
    @info @sprintf "Finished DOS frequency sweep in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (I=ints, E=errs, t=ts, omega=ωs)
end

"""
    run_dos_auto_equispace(HV, Σ::AbstractSelfEnergy, ωs, BZ, rtol, atol, [nthreads=1])

Returns a `NamedTuple` with names `I, E, t, omega, npt1, npt2` containing the
integrals, errors, timings, and number of kpts per dimension used for a density
of states calculation done at frequencies `ωs` with parameters `atol, rtol`.
The domain of integration used is `BZ`.
"""
function run_dos_auto_equispace(H, Σ::AbstractSelfEnergy, ωs, BZ, rtol, atol, nthreads=1)
    test = DOSIntegrand(H, Σ, zero(eltype(ωs)))
    T = AutoBZ.iterated_integral_type(test, BZ)
    ints = Vector{T}(undef, length(ωs))
    errs = Vector{Float64}(undef, length(ωs))
    ts = Vector{Float64}(undef, length(ωs))
    npt1s = Vector{Int}(undef, length(ωs))
    npt2s = Vector{Int}(undef, length(ωs))
    
    @info "Beginning DOS frequency sweep using automatic PTR"
    @info "using $nthreads threads for frequency parallelization"
    t = time()
    batches = batch_smooth_param(ωs, nthreads)
    Threads.@threads for batch in batches
        H_ = deepcopy(H) # to avoid data races for AbstractFourierSeries3D
        # unroll first iteration to get the pre-evaluation buffer
        @info @sprintf "starting ω=%e" batch[1][2]
        t_ = time()
        D = DOSIntegrand(H_, Σ, batch[1][2])
        ints[1], errs[1], rule_buf = automatic_equispace_integration(D, BZ; atol=atol, rtol=rtol)
        ts[1] = time() - t_
        npt1s[1] = rule_buf.npt1
        npt2s[1] = rule_buf.npt2
        for (i, ω) in view(batch, 2:length(batch))
            @info @sprintf "starting ω=%e" ω
            t_ = time()
            D = DOSIntegrand(H_, Σ, ω)
            ints[i], errs[i], rule_buf = automatic_equispace_integration(D, BZ; atol=atol, rtol=rtol, rule_buf...)
            ts[i] = time() - t_
            npt1s[i] = rule_buf.npt1
            npt2s[i] = rule_buf.npt2
            @info @sprintf "finished ω=%e in %e (s) wall clock time" ω ts[i]
        end
    end
    @info @sprintf "Finished DOS frequency sweep in %e (s) wall clock time" (time()-t)
    (I=ints, E=errs, t=ts, npt1=npt1s, npt2=npt2s, omega=ωs)
end

"""
    run_dos_equispace(HV, Σ::AbstractSelfEnergy, ωs, BZ, npt, [nthreads=Threads.nthreads()])

Returns a `NamedTuple` with names `I, t, omega` containing the integrals and
timings obtained for a density of states calculation done at frequencies `ωs`
with parameters `npt`, where `npt` is the number of ``k`` points per
dimension. The caller should check that the result is converged with respect to
`npt`. The domain of integration used is `BZ`.
"""
function run_dos_equispace(H, Σ::AbstractSelfEnergy, ωs, BZ, npt, nthreads=Threads.nthreads())
    test = DOSIntegrand(H, Σ, zero(eltype(ωs)))
    T = AutoBZ.iterated_integral_type(test, BZ)
    ints = Vector{T}(undef, length(ωs))
    ts = Vector{Float64}(undef, length(ωs))
    
    @info "Beginning DOS frequency sweep using PTR with $npt kpts per dim"
    @info "pre-evaluating Hamiltonian..."
    t = time()
    H_k = fourier_pre_eval(H, BZ, npt)
    @info "finished pre-evaluating Hamiltonian in $(time() - t) (s)"
    @info "using $nthreads threads for frequency parallelization"
    t = time()
    batches = batch_smooth_param(ωs, nthreads)
    Threads.@threads for batch in batches
        for (i, ω) in batch
            @info @sprintf "starting ω=%e" ω
            t_ = time()
            D = DOSIntegrand(H, Σ, ω)
            ints[i], = equispace_integration(D, BZ, npt; pre=H_k)
            ts[i] = time() - t_
            @info @sprintf "finished ω=%e in %e (s) wall clock time" ω ts[i]
        end
    end
    @info @sprintf "Finished DOS frequency sweep in %e (s) wall clock time" (time()-t)
    (I=ints, t=ts, omega=ωs)
end


"""
    run_dos(HV, Σ::AbstractSelfEnergy, ωs, BZ, rtol, atol, [nthreads=Threads.nthreads()]; ertol=1.0, eatol=0.0)

Returns a `NamedTuple` with names `I, E, t, omega` containing the integrals,
errors, and timings for a density of states calculation done at frequencies `ωs`
with parameters `atol, rtol`. This function first estimates the integral,
`int`, with an equispace method using wide tolerances `eatol` and `ertol`, and
then uses a narrow absolute tolerance set by `max(atol,rtol*norm(int))` to
evaluate the same integral adaptively. The domain of integration used is `BZ`.

Since this is intended to compute a cheap equispace integral first, it is
recommended to over-ride the default ``k``-grid refinement step to something
``\\eta``-independent with a line like the one below before calling this script

    AutoBZ.equispace_npt_update(npt, ::DOSIntegrand, atol, rtol) = npt + 50
"""
function run_dos(H, Σ::AbstractSelfEnergy, ωs, BZ::AbstractBZ{d}, rtol, atol, nthreads=Threads.nthreads(); ertol=1.0, eatol=0.0, order=7, initdivs=ntuple(i -> Val(1), Val{d}())) where d
    rtol == 0 && return run_dos_adaptive(H, Σ, ωs, BZ, rtol, atol, nthreads; order=order, initdivs=initdivs)
    equi_results = run_dos_auto_equispace(H, Σ, ωs, BZ, ertol, eatol)
    results = run_dos_adaptive(H, Σ, ωs, BZ, 0.0, 0.0, nthreads, [max(atol, rtol*norm(I)) for I in equi_results.I]; order=order, initdivs=initdivs)
    (I=results.I, E=results.E, t=results.t, pre_I=equi_results.I, pre_E=equi_results.E, pre_t=equi_results.t, npt1=equi_results.npt1, npt2=equi_results.npt2, omega=ωs)
end



#=
Section: kinetic coefficient calculations
- 
=#

"""
    get_safe_freq_limits(Ωs, β, lb, ub)

Given a collection of frequencies, `Ωs`, returns a `Vector{CubicLimits{1}}` with
truncated limits of integration for the frequency integral at each `(Ω, β)`
point that are determined by the `fermi_window_limits` routine set to the
default tolerances for the decay of the Fermi window function. The arguments
`lb` and `ub` are lower and upper limits on the frequency to which the default
result gets truncated if the default result would recommend a wider interval. If
there is any truncation, a warning is emitted to the user, but the program will
continue with the truncated limits.
"""
get_safe_freq_limits(Ωs::AbstractVector, β, lb, ub) =
    map(Ω -> get_safe_freq_limits(Ω, β, lb, ub), Ωs)
function get_safe_freq_limits(Ω::Number, β, lb, ub)
    c = fermi_window_limits(Ω, β)
    if (l = only(c.l)) < lb
        @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from below"
        l = lb
    end
    if (u = only(c.u)) > ub
        @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from above"
        u = ub
    end
    CubicLimits(l, u)
end


"Only performs the omega integral"
function test_run_kinetic(HV, Σ::AbstractSelfEnergy, β, n, Ωs, rtol, atol, x, y, z; order=7, initdivs=(Val(1),))
    freq_BZ = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    contract!(HV, z, 3)
    contract!(HV, y, 2)
    contract!(HV, x, 1)
    for (i, (l, Ω)) in enumerate(zip(freq_BZ, Ωs))
        @info @sprintf "starting Ω=%e" Ω
        t = time()
        A = KineticIntegrand(HV, Σ, β, n, Ω)
        ints[i], errs[i] = iterated_integration(A, l; atol=atol, rtol=rtol, order=order, initdivs=initdivs)
        ts[i] = time() - t
        @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
    end
    (I=ints, E=errs, t=ts, Omega=Ωs)
end


"""
    run_kinetic_adaptive(HV, Σ::AbstractSelfEnergy, β, n, Ωs, BZ, rtol, atol, [nthreads=Threads.nthreads()])


Returns a `NamedTuple` with names `I, E, t, Omega` containing the integrals,
errors, and timings for a kinetic coefficient calculation done at frequencies
`Ωs` with parameters `β, n, atol, rtol`. This function constructs a
[`AutoBZ.KineticIntegrand`](@ref) for each parameter value and calls [`AutoBZ.iterated_integration`](@ref) on
it over the domain of the `BZ` and a safely truncated frequency integral to get
the results. The calculation is parallelized over `Ωs` on `nthreads` threads.
"""
function run_kinetic_adaptive(HV, Σ::AbstractSelfEnergy, β, n, Ωs, BZ::AbstractBZ{d}, rtol, atol, nthreads=Threads.nthreads(), atols=fill(atol, length(Ωs)); order=7, initdivs=ntuple(i -> Val(1), Val{d+1}())) where d
    freq_BZ = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    # T = AutoBZ.iterated_integral_type(test, BZ)
    ints = Vector{T}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    
    @info "Beginning OC frequency sweep using IAI"
    @info "using $nthreads threads for frequency parallelization"
    batches = batch_smooth_param(zip(freq_BZ, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        HV_ = deepcopy(HV) # to avoid data races for AbstractFourierSeries3D
        segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, ndims(BZ)+1)
        for (i, (freq_lim, Ω)) in batch
            @info @sprintf "starting Ω=%e" Ω
            t_ = time()
            l = CompositeLimits(BZ, freq_lim)
            A = KineticIntegrand(HV_, Σ, β, n, Ω)
            ints[i], errs[i] = iterated_integration(A, l; atol=atols[i], rtol=rtol, segbufs=segbufs, order=order, initdivs=initdivs)
            ts[i] = time() - t_
            @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
        end
    end
    @info @sprintf "Finished OC frequency sweep in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (I=ints, E=errs, t=ts, Omega=Ωs)
end

"""
    run_kinetic_equispace(HV, Σ::AbstractSelfEnergy, β, n, Ωs, BZ, npt, rtol, atol, [nthreads=Threads.nthreads()])

Returns a `NamedTuple` with names `I, E, t, Omega` containing the
integrals, errors, and timings for a kinetic coefficient calculation done at
frequencies `Ωs` with parameters `β, n, atol, rtol`. This function constructs
an `EquispaceKineticIntegrand` for each parameter value, and precomputes `HV` on
an equispace ``k`` grid with `npt` points per dimension (which is reused for all
parameter values), and calls [`AutoBZ.iterated_integration`](@ref) on it over the domain of the
`BZ` and a safely truncated frequency integral to get the results. The
calculation is parallelized over `Ωs` on `nthreads` threads.
"""
function run_kinetic_equispace(HV, Σ::AbstractSelfEnergy, β, n, Ωs, BZ, npt, rtol, atol, nthreads=Threads.nthreads(); order=7, initdivs=(Val(1),))
    freq_BZ = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    @info "Beginning OC frequency sweep using PTR with $npt kpts per dim"
    @info "pre-evaluating Hamiltonian..."
    t = time()
    pre = fourier_pre_eval(HV, BZ, npt)
    @info "finished pre-evaluating Hamiltonian in $(time() - t) (s)"
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    @info "using $nthreads threads for frequency parallelization"
    batches = batch_smooth_param(zip(freq_BZ, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, 1)
        for (i, (freq_lim, Ω)) in batch
            @info @sprintf "starting Ω=%e" Ω
            t_ = time()
            A = KineticIntegrand(HV, Σ, β, n, Ω)
            EA = EquispaceKineticIntegrand(A, BZ, npt, pre)
            ints[i], errs[i] = iterated_integration(EA, freq_lim; atol=atol, rtol=rtol, segbufs=segbufs, order=order, initdivs=initdivs)
            ts[i] = time() - t_
            @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
        end
    end
    @info @sprintf "Finished OC frequency sweep in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (I=ints, E=errs, t=ts, Omega=Ωs)
end

"""
    run_kinetic_auto_equispace(HV, Σ::AbstractSelfEnergy, β, n, Ωs, BZ, rtol, atol, [nthreads=1])

Returns a `NamedTuple` with names `I, E, t, Omega, npt1, npt2` containing the
integrals, errors, timings, and kpts used  for a kinetic coefficient calculation
done at frequencies `Ωs` with parameters `β, n, atol, rtol`. This function
constructs an `AutoEquispaceKineticIntegrand` for each parameter value, reusing
``k``-grids of `HV` values from previous calculations, and calls
[`AutoBZ.iterated_integration`](@ref) on it over the domain of the `BZ` and a safely truncated
frequency integral to get the results. The calculation is parallelized over `Ωs`
on `nthreads` threads. The default is set to 1 thread for frequency
parallelization, although k-point parallelization is still enabled, to avoid
duplicating calculations of the Hamiltonian and band velocities on the k-mesh.
"""
function run_kinetic_auto_equispace(HV, Σ::AbstractSelfEnergy, β, n, Ωs, BZ, rtol, atol, nthreads=1; order=7, initdivs=(Val(1),))
    freq_BZ = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    npt1s = Vector{Int}(undef, length(Ωs))
    npt2s = Vector{Int}(undef, length(Ωs))
    @info "Beginning OC frequency sweep using automatic PTR"
    @info "using $nthreads threads for frequency parallelization"
    batches = batch_smooth_param(zip(freq_BZ, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        HV_ = deepcopy(HV) # to avoid data races for AbstractFourierSeries3D
        EA = AutoEquispaceKineticIntegrand(KineticIntegrand(HV_, Σ, β, n), BZ, atol, rtol)
        segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, 1)
        for (i, (freq_lim, Ω)) in batch
            @info @sprintf "starting Ω=%e" Ω
            t_ = time()
            EA.A = A = KineticIntegrand(HV_, Σ, β, n, Ω)
            ints[i], errs[i] = iterated_integration(EA, freq_lim; atol=atol, rtol=rtol, order=order, initdivs=initdivs, segbufs=segbufs)
            ts[i] = time() - t_
            npt1s[i] = EA.npt1
            npt2s[i] = EA.npt2
            @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
        end
    end
    @info @sprintf "Finished OC frequency sweep in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (I=ints, E=errs, t=ts, Omega=Ωs, npt1=npt1s, npt2=npt2s)
end


"""
    run_kinetic(HV, Σ::AbstractSelfEnergy, β, n, Ωs, BZ, rtol, atol, [nthreads=Threads.nthreads()]; ertol=1.0, eatol=0.0)

Returns a `NamedTuple` with names `I, E, t, Omega` containing the integrals,
errors, and timings for a kinetic coefficient calculation done at frequencies
`Ωs` with parameters `β, n, atol, rtol`. This function constructs both an
`AutoEquispaceKineticIntegrand` with wide tolerances `eatol` and `ertol` which
estimates the integral, `int`, and then use a narrow absolute tolerance set by
`max(atol,rtol*norm(int))` to construct a [`AutoBZ.KineticIntegrand`](@ref) for each parameter
value and calls [`AutoBZ.iterated_integration`](@ref) on it over the domain of the `BZ` and a
safely truncated frequency integral to get the results. The calculation is
parallelized over `Ωs` on `nthreads` threads.

Since this is intended to compute a cheap equispace integral first, it is
recommended to over-ride the default ``k``-grid refinement step to something
``\\eta``-independent with a line like the one below before calling this script

    AutoBZ.equispace_npt_update(npt, ::TransportIntegrand, atol, rtol) = npt + 50
"""
function run_kinetic(HV, Σ::AbstractSelfEnergy, β, n, Ωs, BZ::AbstractLimits{d}, rtol, atol, nthreads=Threads.nthreads(); ertol=1.0, eatol=0.0, order=7, initdivs=ntuple(i -> Val(1), Val{d+1}())) where d
    rtol == 0 && return run_kinetic_adaptive(HV, Σ, β, n, Ωs, BZ, rtol, atol, nthreads; order=order, initdivs=initdivs)
    equi_results = run_kinetic_auto_equispace(HV, Σ, β, n, Ωs, BZ, ertol, eatol)
    results = run_kinetic_adaptive(HV, Σ, β, n, Ωs, BZ, 0.0, 0.0, nthreads, [max(atol, rtol*norm(I)) for I in equi_results.I]; order=order, initdivs=initdivs)
    (I=results.I, E=results.E, t=results.t, pre_I=equi_results.I, pre_E=equi_results.E, pre_t=equi_results.t, npt1=equi_results.npt1, npt2=equi_results.npt2, Omega=Ωs)
end

# enables kpt parallelization by default for all BZ integrals
function AutoBZ.equispace_evalrule(f::FourierIntegrand, rule, min_per_thread=1, nthreads=Threads.nthreads())
    n = length(rule)
    acc = rule[n][2]*equispace_integrand(f, rule[n][1]) # unroll first term in sum to get right types
    n == 1 && return acc
    runthreads = min(nthreads, div(n-1, min_per_thread)) # choose the actual number of threads
    d, r = divrem(n-1, runthreads)
    partial_sums = fill!(Vector{typeof(acc)}(undef, runthreads), zero(acc)) # allocations :(
    Threads.@threads for i in Base.OneTo(runthreads)
        # batch nodes into `runthreads` continguous groups of size d or d+1 (remainder)
        jmax = (i <= r ? d+1 : d)
        offset = min(i-1, r)*(d+1) + max(i-1-r, 0)*d
        # partial_sums[i] = sum(x -> x[2]*evaluate_integrand(f, x[1]), view(pre, (offset+1):(offset+jmax)); init=zero(acc))
        @inbounds for j in 1:jmax
            partial_sums[i] += rule[offset + j][2]*equispace_integrand(f, rule[offset + j][1])
        end
    end
    # dsum(partial_sums; init=acc)
    for part in partial_sums
        acc += part
    end
    acc
end

# function AutoBZ.equispace_int_eval(f::IteratedFourierIntegrand, pre, dvol, min_per_thread=1, nthreads=Threads.nthreads())
#     error("not implemented")
# end


end # module
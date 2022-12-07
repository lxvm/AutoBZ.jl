module Jobs

using LinearAlgebra
using Printf

using HDF5
using StaticArrays
using OffsetArrays

using ..AutoBZ
using ..AutoBZ.Applications

export read_h5_to_nt, write_nt_to_h5, import_self_energy
export run_dos_parallel, run_dos_auto_parallel
export run_kinetic, run_kinetic_equispace, run_kinetic_auto, run_kinetic_auto_equispace
export run_kinetic_parallel, run_kinetic_equispace_parallel, run_kinetic_auto_parallel, run_kinetic_auto_equispace_parallel

#=
Section: loading data from HDF5
- NamedTuples
- Hamiltonians
- Self Energies
=#

"""
    read_h5_to_nt(filename)

Loads the h5 archive from `filename` and reads its datasets into a `NamedTuple`
"""
read_h5_to_nt(filename) = h5open(filename, "r") do h5
    NamedTuple([(Symbol(key) => h5_dset_to_vec(read(h5, key))) for key in keys(h5)])
end
h5_dset_to_vec(x::Vector) = identity(x)
function h5_dset_to_vec(A::Array{T,N}) where {T,N}
    S = size(A)[1:N-1]
    reinterpret(SArray{Tuple{S...},T,N-1,prod(S)}, vec(A))
end

"""
    import_self_energy(filename)

Reads the groups `omega` and `sigma` in the h5 archive in `filename` and tries
save it to a `NamedTuple` with names `ω` and `Σ`. The array in `sigma` should be
of size `(length(omega), 2)`, where the two columns are the real and imaginary
parts of Σ.
"""
import_self_energy(filename) = h5open(import_self_energy_, filename)
function import_self_energy_(f::HDF5.File)
    dset = read(f, "sigma")
    (ω = read(f, "omega"), Σ = complex.(dset[1, :], dset[2, :]))
end

#=
Section: saving data to HDF5
- NamedTuple to H5
=#

"""
    write_nt_to_h5(nt::NamedTuple, filename)

Takes a `NamedTuple` and writes its values, which must be arrays, into an h5
archive at `filename` with dataset names corresponding to the tuple names.
"""
write_nt_to_h5(nt::NamedTuple, filename) = h5open(filename, "w") do h5
    for key in keys(nt)
        write(h5, string(key), vec_to_h5_dset(nt[key]))
    end
end
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
Section: DOS calculations
- Convergence of PTR at various omega
=#

"""
    run_dos_parallel(H, Σ::AbstractSelfEnergy, μ, ωs, rtol, atol, [nthreads=Threads.nthreads()])

Returns a `NamedTuple` with names `D, err, t, omega` containing the results,
errors, and timings for a density of states calculation done at frequencies `ωs`
with parameters `μ, atol, rtol`. This function constructs a `DOSIntegrand` for
each parameter value and calls `iterated_integration` on it over the domain of
the IBZ to get the results.
"""
function run_dos_parallel(H, Σ::AbstractSelfEnergy, μ, ωs, rtol, atol, nthreads=Threads.nthreads(); order=7, initdivs=(1,1,1))
    T = eltype(DOSIntegrand{typeof(H)})
    BZ_lims = TetrahedralLimits(CubicLimits(period(H)))
    ints = Vector{T}(undef, length(ωs))
    errs = Vector{Float64}(undef, length(ωs))
    ts = Vector{Float64}(undef, length(ωs))
    @info "using $nthreads threads"
    batches = batch_smooth_param(ωs, nthreads)
    t = time()
    Threads.@threads for batch in batches
        H_ = deepcopy(H) # to avoid data races for AbstractFourierSeries3D
        segbufs = AutoBZ.alloc_segbufs(Float64, T, Float64, ndims(BZ_lims))
        for (i, ω) in batch
            @info @sprintf "starting ω=%e" ω
            t_ = time()
            D = DOSIntegrand(H_, ω, Σ, μ)
            ints[i], errs[i] = iterated_integration(D, BZ_lims; atol=atol, rtol=rtol, order=order, initdivs=initdivs, segbufs=segbufs)
            ts[i] = time() - t_
            @info @sprintf "finished ω=%e in %e (s) wall clock time" ω ts[i]
        end
    end
    @info @sprintf "Finished in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (D=ints, err=errs, t=ts, omega=ωs)
end

"""
    run_dos_auto_parallel(HV, Σ::AbstractSelfEnergy, μ, ωs, rtol, atol, [nthreads=Threads.nthreads()]; ertol=1.0, eatol=0.0)

Returns a `NamedTuple` with names `D, err, t, omega` containing the results,
errors, and timings for a density of states calculation done at frequencies `ωs`
with parameters `μ, atol, rtol`. This function first estimates the integral,
`int`, with an equispace method using wide tolerances `eatol` and `ertol`, and
then uses a narrow absolute tolerance set by `max(atol,rtol*norm(int))` to
evaluate the same integral adaptively.

Since this is intended to compute a cheap equispace integral first, it is
recommended to over-ride the default ``k``-grid refinement step to something
``\\eta``-independent with a line like the one below before calling this script

    AutoBZ.equispace_npt_update(npt, ::TransportIntegrand, atol, rtol) = npt + 50
"""
function run_dos_auto_parallel(H, Σ::AbstractSelfEnergy, μ, ωs, rtol, atol, nthreads=Threads.nthreads(); ertol=1.0, eatol=0.0, order=7, initdivs=(1,1,1))
    T = eltype(DOSIntegrand{typeof(H)})
    BZ_lims = TetrahedralLimits(CubicLimits(period(H)))
    ints = Vector{T}(undef, length(ωs))
    errs = Vector{Float64}(undef, length(ωs))
    ts = Vector{Float64}(undef, length(ωs))
    pre_ints = Vector{T}(undef, length(ωs))
    pre_errs = Vector{Float64}(undef, length(ωs))
    pre_ts = Vector{Float64}(undef, length(ωs))
    npt1s = Vector{Int}(undef, length(ωs))
    npt2s = Vector{Int}(undef, length(ωs))
    
    @info "using $nthreads threads"
    t = time()
    @info "Beginning equispace pre-estimate"
    pre_buf = (npt1=0, pre1=Tuple{eltype(H),Int}[], npt2=0, pre2=Tuple{eltype(H),Int}[])
    for (i, ω) in enumerate(ωs)
        @info @sprintf "starting ω=%e" ω
        t_ = time()
        D = DOSIntegrand(H, ω, Σ, μ)
        pre_ints[i], pre_errs[i], pre_buf = automatic_equispace_integration(D, BZ_lims; atol=eatol, rtol=ertol, npt1=pre_buf.npt1, pre1=pre_buf.pre1, npt2=pre_buf.npt2, pre2=pre_buf.pre2)
        pre_ts[i] = time() - t_
        npt1s[i] = pre_buf.npt1
        npt2s[i] = pre_buf.npt2
        @info @sprintf "finished ω=%e in %e (s) wall clock time" ω pre_ts[i]
    end
    @info @sprintf "Finished equispace pre-estimate in %e (s) wall clock time" (time()-t)

    t = time()
    @info "Beginning adaptive integration"
    batches = batch_smooth_param(ωs, nthreads)
    Threads.@threads for batch in batches
        H_ = deepcopy(H) # to avoid data races for AbstractFourierSeries3D
        segbufs = AutoBZ.alloc_segbufs(Float64, T, Float64, ndims(BZ_lims))
        for (i, ω) in batch
            @info @sprintf "starting ω=%e" ω
            t_ = time()
            D = DOSIntegrand(H_, ω, Σ, μ)
            ints[i], errs[i] = iterated_integration(D, BZ_lims; atol=max(atol,rtol*norm(pre_ints[i])), rtol=0.0, order=order, initdivs=initdivs, segbufs=segbufs)
            ts[i] = time() - t_
            @info @sprintf "finished ω=%e in %e (s) wall clock time" ω ts[i]
        end
    end
    @info @sprintf "Finished adaptive integration in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (D=ints, err=errs, t=ts, pre_D=pre_ints, pre_err=pre_errs, pre_t=pre_ts, npt1=npt1s, npt2=npt2s, omega=ωs)
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
function get_safe_freq_limits(Ωs, β, lb, ub)
    freq_lims = Vector{CubicLimits{1,Float64}}(undef, length(Ωs))
    for (i, Ω) in enumerate(Ωs)
        c = fermi_window_limits(Ω, β)
        if (l = only(c.l)) < lb
            @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from below"
            l = lb
        end
        if (u = only(c.u)) > ub
            @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from above"
            u = ub
        end
        freq_lims[i] = CubicLimits(l, u)
    end
    freq_lims
end

"""
    run_kinetic(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol)

Returns a `NamedTuple` with names `A, err, t, Omega` containing the results,
errors, and timings for a kinetic coefficient calculation done at frequencies
`Ωs` with parameters `β, μ, n, atol, rtol`. This function constructs a
`KineticIntegrand` for each parameter value and calls `iterated_integration` on it
over the domain of the IBZ and a safely truncated frequency integral to get the
results.
"""
function run_kinetic(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, ndims(BZ_lims)+1)
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info @sprintf "starting Ω=%e" Ω
        t = time()
        l = CompositeLimits(BZ_lims, freq_lim)
        A = KineticIntegrand(HV, Σ, β, μ, n,  Ω)
        ints[i], errs[i] = iterated_integration(A, l; atol=atol, rtol=rtol, segbufs=segbufs)
        ts[i] = time() - t
        @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
    end
    (A=ints, err=errs, t=ts, Omega=Ωs)
end

"Only performs the omega integral"
function test_run_kinetic(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol, x, y, z)
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    contract!(HV, z, 3)
    contract!(HV, y, 2)
    contract!(HV, x, 1)
    for (i, (l, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info @sprintf "starting Ω=%e" Ω
        t = time()
        A = KineticIntegrand(HV, Σ, β, μ, n, Ω)
        ints[i], errs[i] = iterated_integration(A, l; atol=atol, rtol=rtol)
        ts[i] = time() - t
        @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
    end
    (A=ints, err=errs, t=ts, Omega=Ωs)
end


"""
    run_kinetic_parallel(filename, HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol; nthreads=Threads.nthreads())

Writes an h5 archive to `filename` with groups `A, err, t, Omega` containing
the results, errors, and timings for a kinetic coefficient calculation done at
frequencies `Ωs` with parameters `β, μ, n, atol, rtol`. This function constructs a
`KineticIntegrand` for each parameter value and calls `iterated_integration` on it
over the domain of the IBZ and a safely truncated frequency integral to get the
results. The calculation is parallelized over `Ωs` on `nthreads` threads.
"""
function run_kinetic_parallel(filename, args...; nthreads=Threads.nthreads())
    results = run_kinetic_parallel_(args..., nthreads)
    write_nt_to_h5(results, filename)
    results
end

function run_kinetic_parallel_(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol, nthreads)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    @info "using $nthreads threads"
    batches = batch_smooth_param(zip(freq_lims, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        HV_ = deepcopy(HV) # to avoid data races for AbstractFourierSeries3D
        segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, ndims(BZ_lims)+1)
        for (i, (freq_lim, Ω)) in batch
            @info @sprintf "starting Ω=%e" Ω
            t_ = time()
            l = CompositeLimits(BZ_lims, freq_lim)
            A = KineticIntegrand(HV_, Σ, β, μ, n, Ω)
            ints[i], errs[i] = iterated_integration(A, l; atol=atol, rtol=rtol, segbufs=segbufs)
            ts[i] = time() - t_
            @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
        end
    end
    @info @sprintf "Finished in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (A=ints, err=errs, t=ts, Omega=Ωs)
end

"""
    run_kinetic_equispace(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, npt, rtol, atol; pre_eval=pre_eval_contract)

Returns a `NamedTuple` with names `A, err, t, Omega` containing the results,
errors, and timings for a kinetic coefficient calculation done at frequencies
`Ωs` with parameters `β, μ, n, atol, rtol`. This function constructs an
`EquispaceKineticIntegrand` for each parameter value, and precomputes `HV` on an
equispace ``k`` grid with `npt` points per dimension (which is reused for all
parameter values), and calls `iterated_integration` on it over the domain of the
IBZ and a safely truncated frequency integral to get the results.
"""
function run_kinetic_equispace(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, npt, rtol, atol; pre_eval=pre_eval_contract)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    @info "pre-evaluating Hamiltonian..."
    t = time()
    pre = pre_eval(HV, BZ_lims, npt)
    @info "finished pre-evaluating Hamiltonian in $(time() - t) (s)"
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, 1)
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info @sprintf "starting Ω=%e" Ω
        t = time()
        A = KineticIntegrand(HV, Σ, β, μ, n, Ω)
        EA = EquispaceKineticIntegrand(A, BZ_lims, npt, pre)
        ints[i], errs[i] = iterated_integration(EA, freq_lim; atol=atol, rtol=rtol, segbufs=segbufs)
        ts[i] = time() - t
        @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
    end
    (A=ints, err=errs, t=ts, Omega=Ωs)
end

"""
    run_kinetic_equispace_parallel(filename, HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, npt, rtol, atol; pre_eval=pre_eval_contract, nthreads=Threads.nthreads())

Writes an h5 archive to `filename` with groups `A, err, t, Omega` containing
the results, errors, and timings for a kinetic coefficient calculation done at
frequencies `Ωs` with parameters `β, μ, n, atol, rtol`. This function constructs an
`EquispaceKineticIntegrand` for each parameter value, and precomputes `HV` on an
equispace ``k`` grid with `npt` points per dimension (which is reused for all
parameter values), and calls `iterated_integration` on it over the domain of the
IBZ and a safely truncated frequency integral to get the results. The
calculation is parallelized over `Ωs` on `nthreads` threads.
"""
function run_kinetic_equispace_parallel(filename, args...; pre_eval=pre_eval_contract, nthreads=Threads.nthreads())
    results = run_kinetic_equispace_parallel_(args..., pre_eval, nthreads)
    write_nt_to_h5(results, filename)
    results
end
function run_kinetic_equispace_parallel_(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, npt, rtol, atol, pre_eval, nthreads)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    @info "pre-evaluating Hamiltonian..."
    t = time()
    pre = pre_eval(HV, BZ_lims, npt)
    @info "finished pre-evaluating Hamiltonian in $(time() - t) (s)"
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    @info "using $nthreads threads"
    batches = batch_smooth_param(zip(freq_lims, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, 1)
        for (i, (freq_lim, Ω)) in batch
            @info @sprintf "starting Ω=%e" Ω
            t_ = time()
            A = KineticIntegrand(HV, Σ, β, μ, n, Ω)
            EA = EquispaceKineticIntegrand(A, BZ_lims, npt, pre)
            ints[i], errs[i] = iterated_integration(EA, freq_lim; atol=atol, rtol=rtol, segbufs=segbufs)
            ts[i] = time() - t_
            @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
        end
    end
    @info @sprintf "Finished in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (A=ints, err=errs, t=ts, Omega=Ωs)
end


"""
    run_kinetic_auto(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol; ertol=1.0, eatol=0.0)

Returns a `NamedTuple` with names `A, err, t, Omega` containing the results,
errors, and timings for a kinetic coefficient calculation done at frequencies
`Ωs` with parameters `β, μ, n, atol, rtol`. This function constructs both an
`AutoEquispaceKineticIntegrand` with wide tolerances `eatol` and `ertol` which
estimates the integral, `int`, and then uses a narrow absolute tolerance set by
`max(atol,rtol*norm(int))` to construct a `KineticIntegrand` for each parameter
value and calls `iterated_integration` on it over the domain of the IBZ and a
safely truncated frequency integral to get the results.

Since this is intended to compute a cheap equispace integral first, it is
recommended to over-ride the default ``k``-grid refinement step to something
``\\eta``-independent with a line like the one below before calling this script

    AutoBZ.equispace_npt_update(npt, ::TransportIntegrand, atol, rtol) = npt + 50
"""
function run_kinetic_auto(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol; ertol=1.0, eatol=0.0)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    A = KineticIntegrand(HV, Σ, β, μ, n)
    EA = AutoEquispaceKineticIntegrand(A, BZ_lims, eatol, ertol)
    segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, ndims(BZ_lims)+1)
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info @sprintf "starting Ω=%e" Ω
        t = time()
        l = CompositeLimits(BZ_lims, freq_lim)
        EA.A = A = KineticIntegrand(HV, Σ, β, μ, n, Ω)
        int_, = iterated_integration(EA, freq_lim; atol=eatol, rtol=ertol, segbufs=segbufs)
        atol_ = rtol*norm(int_)
        ints[i], errs[i] = iterated_integration(A, l; atol=max(atol,atol_), rtol=0.0, segbufs=segbufs)
        ts[i] = time() - t
        @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
    end
    (A=ints, err=errs, t=ts, Omega=Ωs)
end


"""
    run_kinetic_auto_parallel(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol; ertol=1.0, eatol=0.0, nthreads=Threads.nthreads())

Returns a `NamedTuple` with names `A, err, t, Omega` containing the results,
errors, and timings for a kinetic coefficient calculation done at frequencies
`Ωs` with parameters `β, μ, n, atol, rtol`. This function constructs both an
`AutoEquispaceKineticIntegrand` with wide tolerances `eatol` and `ertol` which
estimates the integral, `int`, and then use a narrow absolute tolerance set by
`max(atol,rtol*norm(int))` to construct a `KineticIntegrand` for each parameter
value and calls `iterated_integration` on it over the domain of the IBZ and a
safely truncated frequency integral to get the results. The calculation is
parallelized over `Ωs` on `nthreads` threads.

Since this is intended to compute a cheap equispace integral first, it is
recommended to over-ride the default ``k``-grid refinement step to something
``\\eta``-independent with a line like the one below before calling this script

    AutoBZ.equispace_npt_update(npt, ::TransportIntegrand, atol, rtol) = npt + 50
"""
function run_kinetic_auto_parallel(filename, args...; ertol=1.0, eatol=0.0, nthreads=Threads.nthreads())
    results = run_kinetic_auto_parallel_(args..., ertol, eatol, nthreads)
    write_nt_to_h5(results, filename)
    results
end

function run_kinetic_auto_parallel_(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol, ertol, eatol, nthreads)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    pre_ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    pre_errs = Vector{Float64}(undef, length(Ωs))
    pre_ts = Vector{Float64}(undef, length(Ωs))
    npt1s = Vector{Int}(undef, length(Ωs))
    npt2s = Vector{Int}(undef, length(Ωs))
    
    @info "using $nthreads threads"
    t = time()
    @info "Beginning equispace pre-estimate"
    equi_segbuf = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, 1)
    EA = AutoEquispaceKineticIntegrand(KineticIntegrand(HV, Σ, β, μ, n), BZ_lims, eatol, ertol)
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info @sprintf "starting Ω=%e" Ω
        t_ = time()
        EA.A = KineticIntegrand(HV, Σ, β, μ, n, Ω)
        pre_ints[i], pre_errs[i] = iterated_integration(EA, freq_lim; atol=eatol, rtol=ertol, segbufs=equi_segbuf)
        pre_ts[i] = time() - t_
        npt1s[i] = EA.npt1
        npt2s[i] = EA.npt2
        @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω pre_ts[i]
    end
    @info @sprintf "Finished equispace pre-estimate in %e (s) wall clock time" (time()-t)

    t = time()
    @info "Beginning adaptive integration"
    batches = batch_smooth_param(zip(freq_lims, Ωs), nthreads)
    Threads.@threads for batch in batches
        HV_ = deepcopy(HV) # to avoid data races for AbstractFourierSeries3D
        segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, ndims(BZ_lims)+1)
        for (i, (freq_lim, Ω)) in batch
            @info @sprintf "starting Ω=%e" Ω
            t_ = time()
            l = CompositeLimits(BZ_lims, freq_lim)
            A = KineticIntegrand(HV_, Σ, β, μ, n, Ω)
            ints[i], errs[i] = iterated_integration(A, l; atol=max(atol,rtol*norm(pre_ints[i])), rtol=0.0, segbufs=segbufs)
            ts[i] = time() - t_
            @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
        end
    end
    @info @sprintf "Finished adaptive integration in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (A=ints, err=errs, t=ts, pre_A=pre_ints, pre_err=pre_errs, pre_t=pre_ts, npt1=npt1s, npt2=npt2s, Omega=Ωs)
end

"""
    run_kinetic_auto_equispace(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol)

Returns a `NamedTuple` with names `A, err, t, Omega` containing the results,
errors, and timings for a kinetic coefficient calculation done at frequencies
`Ωs` with parameters `β, μ, n, atol, rtol`. This function constructs an
`AutoEquispaceKineticIntegrand` for each parameter value, reusing ``k``-grids of `HV`
values from previous calculations, and calls `iterated_integration` on it over
the domain of the IBZ and a safely truncated frequency integral to get the
results.
"""
function run_kinetic_auto_equispace(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    A = KineticIntegrand(HV, Σ, β, μ, n)
    EA = AutoEquispaceKineticIntegrand(A, BZ_lims, atol, rtol)
    segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, 1)
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info @sprintf "starting Ω=%e" Ω
        t = time()
        EA.A = A = KineticIntegrand(HV, Σ, β, μ, n, Ω)
        ints[i], errs[i] = iterated_integration(EA, freq_lim; atol=atol, rtol=rtol, segbufs=segbufs)
        ts[i] = time() - t
        @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
    end
    (A=ints, err=errs, t=ts, Omega=Ωs)
end

"""
    run_kinetic_auto_equispace_parallel(filename, HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol; nthreads=1)

Writes an h5 archive to `filename` with groups `A, err, t, Omega` containing
the results, errors, and timings for a kinetic coefficient calculation done at
frequencies `Ωs` with parameters `β, μ, n, atol, rtol`. This function constructs an
`AutoEquispaceKineticIntegrand` for each parameter value, reusing ``k``-grids of `HV`
values from previous calculations, and calls `iterated_integration` on it
over the domain of the IBZ and a safely truncated frequency integral to get the
results. The calculation is parallelized over `Ωs` on `nthreads` threads. The
default is set to 1 thread for frequency parallelization, although k-point
parallelization is still enabled, to avoid duplicating calculations of the
Hamiltonian and band velocities on the k-mesh.
"""
function run_kinetic_auto_equispace_parallel(filename, args...; nthreads=1)
    results = run_kinetic_auto_equispace_parallel_(args..., nthreads)
    write_nt_to_h5(results, filename)
    results
end

function run_kinetic_auto_equispace_parallel_(HV, Σ::AbstractSelfEnergy, β, μ, n, Ωs, rtol, atol, nthreads)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    @info "using $nthreads threads"
    batches = batch_smooth_param(zip(freq_lims, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        HV_ = deepcopy(HV) # to avoid data races for AbstractFourierSeries3D
        EA = AutoEquispaceKineticIntegrand(KineticIntegrand(HV_, Σ, β, μ, n), BZ_lims, atol, rtol)
        segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, 1)
        for (i, (freq_lim, Ω)) in batch
            @info @sprintf "starting Ω=%e" Ω
            t_ = time()
            EA.A = A = KineticIntegrand(HV_, Σ, β, μ, n, Ω)
            ints[i], errs[i] = iterated_integration(EA, freq_lim; atol=atol, rtol=rtol, segbufs=segbufs)
            ts[i] = time() - t_
            @info @sprintf "finished Ω=%e in %e (s) wall clock time" Ω ts[i]
        end
    end
    @info @sprintf "Finished in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (A=ints, err=errs, t=ts, Omega=Ωs)
end

# enables kpt parallelization by default for transport and kinetic integrals
function AutoBZ.equispace_int_eval(f::TransportIntegrand, pre, dvol; min_per_thread=1, nthreads=Threads.nthreads())
    n = length(pre)
    acc = pre[n][2]*evaluate_integrand(f, pre[n][1]) # unroll first term in sum to get right types
    runthreads = min(nthreads, div(n-1, min_per_thread)) # choose the actual number of threads
    d, r = divrem(n-1, runthreads)
    partial_sums = zeros(typeof(acc), runthreads) # allocations :(
    Threads.@threads for i in Base.OneTo(runthreads)
        # batch nodes into `runthreads` continguous groups of size d or d+1 (remainder)
        jmax = (i <= r ? d+1 : d)
        offset = min(i-1, r)*(d+1) + max(i-1-r, 0)*d
        # partial_sums[i] = sum(x -> x[2]*evaluate_integrand(f, x[1]), view(pre, (offset+1):(offset+jmax)); init=zero(acc))
        @inbounds for j in 1:jmax
            x, w = pre[offset + j]
            partial_sums[i] += w*evaluate_integrand(f, x)
        end
    end
    # dvol*sum(partial_sums; init=acc)
    for part in partial_sums
        acc += part
    end
    dvol*acc
end

end # module
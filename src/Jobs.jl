module Jobs

using LinearAlgebra

using HDF5
using StaticArrays
using OffsetArrays

using ..AutoBZ
using ..AutoBZ.Applications

export BandEnergyBerryVelocities
export read_h5_to_nt, write_nt_to_h5, import_self_energy
export OCscript, OC_script_equispace, OC_script_auto, OC_script_auto_equispace
export OCscript_parallel, OC_script_equispace_parallel, OC_script_auto_parallel, OC_script_auto_equispace_parallel

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



#=
Section: OC calculations
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
    BandEnergyBerryVelocities

Union type of `BandEnergyBerryVelocity`, `BandEnergyBerryVelocity3D`,
`BandEnergyVelocity`, and `BandEnergyVelocity3D`.
"""
const BandEnergyBerryVelocities = Union{BandEnergyBerryVelocity,BandEnergyBerryVelocity3D,BandEnergyVelocity,BandEnergyVelocity3D}

"""
    OCscript(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol)

Returns a `NamedTuple` with names `OC, err, t, Omega` containing the results,
errors, and timings for an optical conductivity calculation done at frequencies
`Ωs` with parameters `β, μ, atol, rtol`. This function constructs an
`OCIntegrand` for each parameter value and calls `iterated_integration` on it
over the domain of the IBZ and a safely truncated frequency integral to get the
results.
"""
function OCscript(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    segbufs = AutoBZ.alloc_segbufs(Float64, eltype(OCIntegrand), Float64, ndims(BZ_lims)+1)
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info "Ω=$Ω starting ..."
        t = time()
        l = CompositeLimits(BZ_lims, freq_lim)
        σ = OCIntegrand(HV, Σ, Ω, β, μ)
        ints[i], errs[i] = iterated_integration(σ, l; atol=atol, rtol=rtol, segbufs=segbufs)
        ts[i] = time() - t
        @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
    end
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end

"Only performs the omega integral"
function test_OCscript(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol, x, y, z)
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    ν₁ = FourierSeriesDerivative(H, SVector(1,0,0))
    ν₂ = FourierSeriesDerivative(H, SVector(0,1,0))
    ν₃ = FourierSeriesDerivative(H, SVector(0,0,1))
    H_ = contract(contract(contract(HV, z), y), x)
    ν₁_ = contract(contract(contract(ν₁, z), y), x)
    ν₂_ = contract(contract(contract(ν₂, z), y), x)
    ν₃_ = contract(contract(contract(ν₃, z), y), x)
    for (i, (l, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info "Ω=$Ω starting ..."
        t = time()
        σ = OCIntegrand(H_,ν₁_, ν₂_, ν₃_, Σ, Ω, β, μ)
        ints[i], errs[i] = iterated_integration(σ, l; atol=atol, rtol=rtol)
        ts[i] = time() - t
        @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
    end
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end


"""
    OCscript_parallel(filename, HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol; nthreads=Threads.nthreads())

Writes an h5 archive to `filename` with groups `OC, err, t, Omega` containing
the results, errors, and timings for an optical conductivity calculation done at
frequencies `Ωs` with parameters `β, μ, atol, rtol`. This function constructs an
`OCIntegrand` for each parameter value and calls `iterated_integration` on it
over the domain of the IBZ and a safely truncated frequency integral to get the
results. The calculation is parallelized over `Ωs` on `nthreads` threads.
"""
function OCscript_parallel(filename, args...; nthreads=Threads.nthreads())
    results = OCscript_parallel_(args..., nthreads)
    write_nt_to_h5(results, filename)
    results
end

function OCscript_parallel_(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol, nthreads)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    @info "using $nthreads threads"
    batches = batch_smooth_param(zip(freq_lims, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        segbufs = AutoBZ.alloc_segbufs(Float64, eltype(OCIntegrand), Float64, ndims(BZ_lims)+1)
        for (i, (freq_lim, Ω)) in batch
            @info "Ω=$Ω started"
            t_ = time()
            l = CompositeLimits(BZ_lims, freq_lim)
            σ = OCIntegrand(HV, Σ, Ω, β, μ)
            ints[i], errs[i] = iterated_integration(σ, l; atol=atol, rtol=rtol, segbufs=segbufs)
            ts[i] = time() - t_
            @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
        end
    end
    @info "Finished in $(sum(ts)) (s) CPU time and $(time()-t) (s) wall clock time"
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end

"""
    OCscript_equispace(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, npt, atol, rtol; pre_eval=pre_eval_contract)

Returns a `NamedTuple` with names `OC, err, t, Omega` containing the results,
errors, and timings for an optical conductivity calculation done at frequencies
`Ωs` with parameters `β, μ, atol, rtol`. This function constructs an
`EquispaceOCIntegrand` for each parameter value, and precomputes `HV` on an
equispace ``k`` grid with `npt` points per dimension (which is reused for all
parameter values), and calls `iterated_integration` on it over the domain of the
IBZ and a safely truncated frequency integral to get the results.
"""
function OCscript_equispace(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, npt, atol, rtol; pre_eval=pre_eval_contract)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    @info "pre-evaluating Hamiltonian..."
    t = time()
    pre = pre_eval(HV, BZ_lims, npt)
    @info "finished pre-evaluating Hamiltonian in $(time() - t) (s)"
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    segbufs = AutoBZ.alloc_segbufs(Float64, eltype(OCIntegrand), Float64, 1)
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info "Ω=$Ω starting ..."
        t = time()
        σ = OCIntegrand(HV, Σ, Ω, β, μ)
        Eσ = EquispaceOCIntegrand(σ, BZ_lims, npt, pre)
        ints[i], errs[i] = iterated_integration(Eσ, freq_lim; atol=atol, rtol=rtol, segbufs=segbufs)
        ts[i] = time() - t
        @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
    end
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end

"""
    OCscript_equispace_parallel(filename, HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, npt, atol, rtol; pre_eval=pre_eval_contract, nthreads=Threads.nthreads())

Writes an h5 archive to `filename` with groups `OC, err, t, Omega` containing
the results, errors, and timings for an optical conductivity calculation done at
frequencies `Ωs` with parameters `β, μ, atol, rtol`. This function constructs an
`EquispaceOCIntegrand` for each parameter value, and precomputes `HV` on an
equispace ``k`` grid with `npt` points per dimension (which is reused for all
parameter values), and calls `iterated_integration` on it over the domain of the
IBZ and a safely truncated frequency integral to get the results. The
calculation is parallelized over `Ωs` on `nthreads` threads.
"""
function OCscript_equispace_parallel(filename, args...; pre_eval=pre_eval_contract, nthreads=Threads.nthreads())
    results = OCscript_equispace_parallel_(args..., pre_eval, nthreads)
    write_nt_to_h5(results, filename)
    results
end
function OCscript_equispace_parallel_(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, npt, atol, rtol, pre_eval, nthreads)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    @info "pre-evaluating Hamiltonian..."
    t = time()
    pre = pre_eval(HV, BZ_lims, npt)
    @info "finished pre-evaluating Hamiltonian in $(time() - t) (s)"
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    @info "using $nthreads threads"
    batches = batch_smooth_param(zip(freq_lims, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        segbufs = AutoBZ.alloc_segbufs(Float64, eltype(OCIntegrand), Float64, 1)
        for (i, (freq_lim, Ω)) in batch
            @info "Ω=$Ω starting ..."
            t_ = time()
            σ = OCIntegrand(HV, Σ, Ω, β, μ)
            Eσ = EquispaceOCIntegrand(σ, BZ_lims, npt, pre)
            ints[i], errs[i] = iterated_integration(Eσ, freq_lim; atol=atol, rtol=rtol, segbufs=segbufs)
            ts[i] = time() - t_
            @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
        end
    end
    @info "Finished in $(sum(ts)) (s) CPU time and $(time()-t) (s) wall clock time"
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end


"""
    OCscript_auto(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol; ertol=1.0, eatol=0.0)

Returns a `NamedTuple` with names `OC, err, t, Omega` containing the results,
errors, and timings for an optical conductivity calculation done at frequencies
`Ωs` with parameters `β, μ, atol, rtol`. This function constructs both an
`AutoEquispaceOCIntegrand` with wide tolerances `eatol` and `ertol` which
estimates the integral, `int`, and then uses a narrow absolute tolerance set by
`max(atol,rtol*norm(int))` to construct an `OCIntegrand` for each parameter
value and calls `iterated_integration` on it over the domain of the IBZ and a
safely truncated frequency integral to get the results.

Since this is intended to compute a cheap equispace integral first, it is
recommended to over-ride the default ``k``-grid refinement step to something
``\\eta``-independent with a line like the one below before calling this script

    AutoBZ.equispace_npt_update(npt, ::GammaIntegrand, atol, rtol) = npt + 50
"""
function OCscript_auto(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, rtol, atol; ertol=1.0, eatol=0.0)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    σ = OCIntegrand(HV, Σ, 0.0, β, μ)
    Eσ = AutoEquispaceOCIntegrand(σ, BZ_lims, eatol, ertol)
    segbufs = AutoBZ.alloc_segbufs(Float64, eltype(OCIntegrand), Float64, ndims(BZ_lims)+1)
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info "Ω=$Ω starting ..."
        t = time()
        l = CompositeLimits(BZ_lims, freq_lim)
        Eσ.σ = σ = OCIntegrand(HV, Σ, Ω, β, μ)
        int_, = iterated_integration(Eσ, freq_lim; atol=eatol, rtol=ertol, segbufs=segbufs)
        atol_ = rtol*norm(int_)
        ints[i], errs[i] = iterated_integration(σ, l; atol=max(atol,atol_), rtol=0.0, segbufs=segbufs)
        ts[i] = time() - t
        @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
    end
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end


"""
    OCscript_auto_parallel(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol; ertol=1.0, eatol=0.0, nthreads=Threads.nthreads())

Returns a `NamedTuple` with names `OC, err, t, Omega` containing the results,
errors, and timings for an optical conductivity calculation done at frequencies
`Ωs` with parameters `β, μ, atol, rtol`. This function constructs both an
`AutoEquispaceOCIntegrand` with wide tolerances `eatol` and `ertol` which
estimates the integral, `int`, and then use a narrow absolute tolerance set by
`max(atol,rtol*norm(int))` to construct an `OCIntegrand` for each parameter
value and calls `iterated_integration` on it over the domain of the IBZ and a
safely truncated frequency integral to get the results. The calculation is
parallelized over `Ωs` on `nthreads` threads.

Since this is intended to compute a cheap equispace integral first, it is
recommended to over-ride the default ``k``-grid refinement step to something
``\\eta``-independent with a line like the one below before calling this script

    AutoBZ.equispace_npt_update(npt, ::GammaIntegrand, atol, rtol) = npt + 50
"""
function OCscript_auto_parallel(filename, args...; ertol=1.0, eatol=0.0, nthreads=Threads.nthreads())
    results = OCscript_auto_parallel_(args..., ertol, eatol, nthreads)
    write_nt_to_h5(results, filename)
    results
end

function OCscript_auto_parallel_(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, rtol, atol, ertol, eatol, nthreads)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    pre_ts = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    @info "using $nthreads threads"
    batches = batch_smooth_param(zip(freq_lims, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        σ = OCIntegrand(HV, Σ, 0.0, β, μ)
        Eσ = AutoEquispaceOCIntegrand(σ, BZ_lims, eatol, ertol)
        segbufs = AutoBZ.alloc_segbufs(Float64, eltype(OCIntegrand), Float64, ndims(BZ_lims)+1)
        for (i, (freq_lim, Ω)) in batch
            @info "Ω=$Ω starting ..."
            t_ = time()
            l = CompositeLimits(BZ_lims, freq_lim)
            Eσ.σ = σ = OCIntegrand(HV, Σ, Ω, β, μ)
            int_, = iterated_integration(Eσ, freq_lim; atol=eatol, rtol=ertol, segbufs=segbufs)
            atol_ = rtol*norm(int_)
            pre_ts[i] = time() - t_
            t_ = time()
            ints[i], errs[i] = iterated_integration(σ, l; atol=max(atol,atol_), rtol=0.0, segbufs=segbufs)
            ts[i] = time() - t_
            @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
        end
    end
    @info "Finished in $(sum(ts)+sum(pre_ts)) (s) CPU time and $(time()-t) (s) wall clock time"
    (OC=ints, err=errs, t=ts, pre_t=pre_ts, Omega=Ωs)
end

"""
    OCscript_auto_equispace(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol)

Returns a `NamedTuple` with names `OC, err, t, Omega` containing the results,
errors, and timings for an optical conductivity calculation done at frequencies
`Ωs` with parameters `β, μ, atol, rtol`. This function constructs an
`AutoEquispaceOCIntegrand` for each parameter value, reusing ``k``-grids of `HV`
values from previous calculations, and calls `iterated_integration` on it over
the domain of the IBZ and a safely truncated frequency integral to get the
results.
"""
function OCscript_auto_equispace(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, rtol, atol)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    σ = OCIntegrand(HV, Σ, 0.0, β, μ)
    Eσ = AutoEquispaceOCIntegrand(σ, BZ_lims, atol, rtol)
    segbufs = AutoBZ.alloc_segbufs(Float64, eltype(OCIntegrand), Float64, 1)
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info "Ω=$Ω starting ..."
        t = time()
        Eσ.σ = σ = OCIntegrand(HV, Σ, Ω, β, μ)
        ints[i], errs[i] = iterated_integration(Eσ, freq_lim; atol=atol, rtol=rtol, segbufs=segbufs)
        ts[i] = time() - t
        @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
    end
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end

"""
    OCscript_auto_equispace_parallel(filename, HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol; nthreads=Threads.nthreads())

Writes an h5 archive to `filename` with groups `OC, err, t, Omega` containing
the results, errors, and timings for an optical conductivity calculation done at
frequencies `Ωs` with parameters `β, μ, atol, rtol`. This function constructs an
`AutoEquispaceOCIntegrand` for each parameter value, reusing ``k``-grids of `HV`
values from previous calculations, and calls `iterated_integration` on it
over the domain of the IBZ and a safely truncated frequency integral to get the
results. The calculation is parallelized over `Ωs` on `nthreads` threads.
"""
function OCscript_auto_equispace_parallel(filename, args...; nthreads=Threads.nthreads())
    results = OCscript_auto_equispace_parallel_(args..., nthreads)
    write_nt_to_h5(results, filename)
    results
end

function OCscript_auto_equispace_parallel_(HV::BandEnergyBerryVelocities, Σ::AbstractSelfEnergy, β, Ωs, μ, rtol, atol, nthreads)
    BZ_lims = TetrahedralLimits(CubicLimits(period(HV)))
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    @info "using $nthreads threads"
    batches = batch_smooth_param(zip(freq_lims, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        σ = OCIntegrand(HV, Σ, 0.0, β, μ)
        Eσ = AutoEquispaceOCIntegrand(σ, BZ_lims, atol, rtol)
        segbufs = AutoBZ.alloc_segbufs(Float64, eltype(OCIntegrand), Float64, 1)
        for (i, (freq_lim, Ω)) in batch
            @info "Ω=$Ω starting ..."
            t_ = time()
            Eσ.σ = σ = OCIntegrand(HV, Σ, Ω, β, μ)
            ints[i], errs[i] = iterated_integration(Eσ, freq_lim; atol=atol, rtol=rtol, segbufs=segbufs)
            ts[i] = time() - t_
            @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
        end
    end
    @info "Finished in $(sum(ts)) (s) CPU time and $(time()-t) (s) wall clock time"
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end

end # module
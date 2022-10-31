module Demos

using LinearAlgebra

using HDF5
using StaticArrays
using OffsetArrays

using AutoBZ
using AutoBZ.Applications

#=
Section: loading data from HDF5
- NamedTuples
- Hamiltonians
- Self Energies
=#

read_h5_to_nt(filename) = h5open(filename, "r") do h5
    NamedTuple([(Symbol(key) => h5_dset_to_vec(read(h5, key))) for key in keys(h5)])
end
h5_dset_to_vec(x::Vector) = identity(x)
function h5_dset_to_vec(A::Array{T,N}) where {T,N}
    S = size(A)[1:N-1]
    reinterpret(SArray{Tuple{S...},T,N-1,prod(S)}, vec(A))
end

loadW90Hamiltonian(filename) = h5open(loadW90Hamiltonian_, filename)

function loadW90Hamiltonian_(f::HDF5.File)
    dset = f["epsilon_mn"]
    idxs = f["irvec"]
    A = read(dset)
    s = Int(cbrt(size(A, 3)))
    k = Int((s-1)/2)
    idx = read(idxs)
    T = SMatrix{size(A,1), size(A,2), ComplexF64, size(A,1)*size(A,2)}
    # M = LinearAlgebra.checksquare(A[:,:,1,1])
    # T = SHermitianCompact{M,ComplexF64,StaticArrays.triangularnumber(M)}
    C = loadC(T, A, idx .+ (k+1), (s,s,s))
    C = OffsetArray(C, -k:k, -k:k, -k:k)
    C
end

function loadC(T, A::Array{Float64,4}, idx, dims)
    C = Array{T, length(dims)}(undef, dims)
    for i in axes(A, 3)
        C[CartesianIndex(idx[:, i]...)] = T(complex.(view(A, :, :, i, 1), view(A, :, :, i, 2)))
    end
    C
end

import_self_energy(filename) = h5open(import_self_energy_, filename)
function import_self_energy_(f::HDF5.File)
    dset = read(f, "sigma")
    (ω = read(f, "omega"), Σ = complex.(dset[1, :], dset[2, :]))
end

self_energy_evaluator(filename, order) = self_energy_evaluator(import_self_energy(filename)..., order)
function self_energy_evaluator(ω, Σ, order)
    Σ = self_energy_interpolant(ω, Σ, order)
    Applications.ScalarEnergy(Σ, only(lb(Σ)), only(ub(Σ)))
end
self_energy_interpolant(ω, Σ, order) = chebregression(ω, Σ, (order,))

#=
Section: saving data to HDF5
- NamedTuple to H5
=#

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
If the cost of a calculation smoothly varies with the parameters `xs`, then
batch `xs` into `nthreads` groups where the `i`th element of group `j` is
`xs[j+(i-1)*nthreads]`
"""
function batch_smooth_param(xs, nthreads)
    batches = [Tuple{Int,eltype(xs)}[] for _ in 1:nthreads]
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

function get_safe_freq_limits(Ωs, β, lb, ub)
    freq_lims = Vector{CubicLimits{1,Float64}}(undef, length(Ωs))
    for (i, Ω) in enumerate(Ωs)
        c = Applications.fermi_window_limits(Ω, β)
        l = only(c.l)
        u = only(c.u)
        if l < lb
            @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from below"
            l = lb
        end
        if u > ub
            @warn "At Ω=$Ω, β=$β, the interpolant limits the desired frequency window from above"
            u = ub
        end
        freq_lims[i] = CubicLimits(l, u)
    end
    freq_lims
end

"Performs the full calculation"
function OCscript(H::FourierSeries, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol)
    BZ_lims = TetrahedralLimits(H.period)
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info "Ω=$Ω starting ..."
        t = time()
        l = CompositeLimits(BZ_lims, freq_lim)
        σ = OCIntegrand(H, Σ, Ω, β, μ)
        ints[i], errs[i] = iterated_integration(σ, l; atol=atol, rtol=rtol, callback=contract)
        ts[i] = time() - t
        @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
    end
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end

"Only performs the omega integral"
function test_OCscript(H::FourierSeries, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol, x, y, z)
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    ν₁ = FourierSeriesDerivative(H, SVector(1,0,0))
    ν₂ = FourierSeriesDerivative(H, SVector(0,1,0))
    ν₃ = FourierSeriesDerivative(H, SVector(0,0,1))
    H_ = contract(contract(contract(H, z), y), x)
    ν₁_ = contract(contract(contract(ν₁, z), y), x)
    ν₂_ = contract(contract(contract(ν₂, z), y), x)
    ν₃_ = contract(contract(contract(ν₃, z), y), x)
    for (i, (l, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info "Ω=$Ω starting ..."
        t = time()
        σ = OCIntegrand(H_,ν₁_, ν₂_, ν₃_, Σ, Ω, β, μ)
        ints[i], errs[i] = iterated_integration(σ, l; atol=atol, rtol=rtol, callback=contract)
        ts[i] = time() - t
        @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
    end
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end

function OCscript_parallel(filename, args...)
    results = OCscript_parallel_(args...)
    write_nt_to_h5(results, filename)
    results
end

function OCscript_parallel_(H::FourierSeries, Σ::AbstractSelfEnergy, β, Ωs, μ, atol, rtol)
    BZ_lims = TetrahedralLimits(H.period)
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    nthreads = Threads.nthreads()
    @info "using $nthreads threads"
    batches = batch_smooth_param(zip(freq_lims, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        for (i, (freq_lim, Ω)) in batch
            @info "Ω=$Ω started"
            t_ = time()
            l = CompositeLimits(BZ_lims, freq_lim)
            σ = OCIntegrand(H, Σ, Ω, β, μ)
            ints[i], errs[i] = iterated_integration(σ, l; atol=atol, rtol=rtol, callback=contract)
            ts[i] = time() - t_
            @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
        end
    end
    @info "Finished in $(sum(ts)) (s) CPU time and $(time()-t) (s) wall clock time"
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end

function OCscript_equispace(H::FourierSeries, Σ::AbstractSelfEnergy, β, Ωs, μ, npt, atol, rtol; pre_eval=pre_eval_contract)
    BZ_lims = TetrahedralLimits(H.period)
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    HV = BandEnergyVelocity(H)
    @info "pre-evaluating Hamiltonian..."
    t = time()
    pre = pre_eval(HV, BZ_lims, npt)
    @info "finished pre-evaluating Hamiltonian in $(time() - t) (s)"
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info "Ω=$Ω starting ..."
        t = time()
        σ = OCIntegrand(HV, Σ, Ω, β, μ)
        Eσ = EquispaceOCIntegrand(σ, BZ_lims, npt, pre)
        ints[i], errs[i] = iterated_integration(Eσ, freq_lim; atol=atol, rtol=rtol, callback=contract)
        ts[i] = time() - t
        @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
    end
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end

function OCscript_equispace_parallel(filename, args...)
    results = OCscript_equispace_parallel_(args...)
    write_nt_to_h5(results, filename)
    results
end
function OCscript_equispace_parallel_(H::FourierSeries, Σ::AbstractSelfEnergy, β, Ωs, μ, npt, atol, rtol; pre_eval=pre_eval_contract)
    BZ_lims = TetrahedralLimits(H.period)
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    HV = BandEnergyVelocity(H)
    @info "pre-evaluating Hamiltonian..."
    t = time()
    pre = pre_eval(HV, BZ_lims, npt)
    @info "finished pre-evaluating Hamiltonian in $(time() - t) (s)"
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    nthreads = Threads.nthreads()
    @info "using $nthreads threads"
    batches = batch_smooth_param(zip(freq_lims, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        for (i, (freq_lim, Ω)) in batch
            @info "Ω=$Ω starting ..."
            t_ = time()
            σ = OCIntegrand(HV, Σ, Ω, β, μ)
            Eσ = EquispaceOCIntegrand(σ, BZ_lims, npt, pre)
            ints[i], errs[i] = iterated_integration(Eσ, freq_lim; atol=atol, rtol=rtol, callback=contract)
            ts[i] = time() - t_
            @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
        end
    end
    @info "Finished in $(sum(ts)) (s) CPU time and $(time()-t) (s) wall clock time"
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end

function OCscript_auto(H::FourierSeries, Σ::AbstractSelfEnergy, β, Ωs, μ, rtol; ertol=1.0, eatol=0.0)
    BZ_lims = TetrahedralLimits(H.period)
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    σ = OCIntegrand(H, Σ, 0.0, β, μ)
    Eσ = AutoEquispaceOCIntegrand(σ, BZ_lims, eatol, ertol)
    for (i, (freq_lim, Ω)) in enumerate(zip(freq_lims, Ωs))
        @info "Ω=$Ω starting ..."
        t = time()
        l = CompositeLimits(BZ_lims, freq_lim)
        Eσ.σ = σ = OCIntegrand(H, Σ, Ω, β, μ)
        int_, = iterated_integration(Eσ, freq_lim; atol=eatol, rtol=ertol)
        atol = rtol*10^floor(log10(norm(int_)))
        ints[i], errs[i] = iterated_integration(σ, l; atol=atol, rtol=0.0, callback=contract)
        ts[i] = time() - t
        @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
    end
    (OC=ints, err=errs, t=ts, Omega=Ωs)
end


function OCscript_auto_parallel(filename, args...)
    results = OCscript_auto_parallel_(args...)
    write_nt_to_h5(results, filename)
    results
end

function OCscript_auto_parallel_(H::FourierSeries, Σ::AbstractSelfEnergy, β, Ωs, μ, rtol, atol; ertol=1.0, eatol=0.0)
    BZ_lims = TetrahedralLimits(H.period)
    freq_lims = get_safe_freq_limits(Ωs, β, lb(Σ), ub(Σ))
    ints = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
    errs = Vector{Float64}(undef, length(Ωs))
    pre_ts = Vector{Float64}(undef, length(Ωs))
    ts = Vector{Float64}(undef, length(Ωs))
    nthreads = Threads.nthreads()
    @info "using $nthreads threads"
    batches = batch_smooth_param(zip(freq_lims, Ωs), nthreads)
    t = time()
    Threads.@threads for batch in batches
        σ = OCIntegrand(H, Σ, 0.0, β, μ)
        Eσ = AutoEquispaceOCIntegrand(σ, BZ_lims, eatol, ertol)
        for (i, (freq_lim, Ω)) in batch
            @info "Ω=$Ω starting ..."
            t_ = time()
            l = CompositeLimits(BZ_lims, freq_lim)
            Eσ.σ = σ = OCIntegrand(H, Σ, Ω, β, μ)
            int_, = iterated_integration(Eσ, freq_lim; atol=eatol, rtol=ertol)
            atol_ = rtol*10^floor(log10(norm(int_)))
            pre_ts[i] = time() - t_
            t_ = time()
            ints[i], errs[i] = iterated_integration(σ, l; atol=max(atol,atol_), rtol=0.0, callback=contract)
            ts[i] = time() - t_
            @info "Ω=$Ω finished in $(ts[i]) (s) wall clock time"
        end
    end
    @info "Finished in $(sum(ts)+sum(pre_ts)) (s) CPU time and $(time()-t) (s) wall clock time"
    (OC=ints, err=errs, t=ts, pre_t=pre_ts, Omega=Ωs)
end

end # module
# hdf5

"""
    read_h5_to_nt(filename)

Loads the h5 archive from `filename` and reads its datasets into a `NamedTuple`
and its groups into `NamedTuple`s recursively.
"""
read_h5_to_nt(filename) = h5open(read_h5_to_nt_, filename, "r")
read_h5_to_nt_(h5) = NamedTuple([Pair(Symbol(key), ((val = h5[key]) isa HDF5.Group) ? read_h5_to_nt_(val) : h5_dset_to_vec(read(h5, key))) for key in keys(h5)])
h5_dset_to_vec(x::Array) = x
#=
function h5_dset_to_vec(A::Array{T,N}) where {T,N}
    S = size(A)[1:N-1]
    reinterpret(SArray{Tuple{S...},T,N-1,prod(S)}, vec(A))
end
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


# parallelization

batchsolve(s::String, f, ps, T=Base.promote_op(f, eltype(ps)); mode="w", kwargs...) = h5open(s, mode) do h5
    dims = dataspace(tuple(length(ps)))
    F, fdims = if T <: Number
        datatype(T), dims
    elseif T <: SArray
        datatype(eltype(T)), dataspace((size(ps)..., size(T)...))
    else
        throw(ArgumentError("Couldn't map result type to HDF5 type"))
    end
    gI = create_dataset(h5, "I", F, fdims)
    gE = create_dataset(h5, "E", datatype(Float64), dims)
    gt = create_dataset(h5, "t", datatype(Float64), dims)
    gp = create_dataset(h5, "p", datatype(eltype(ps)), dims)
    ax = ntuple(_-> :, Val(ndims(T)))

    function h5callback(f, i, p, sol, t)
        @info @sprintf "parameter %i finished in %e (s)" i t
        gI[i,ax...] = sol.u
        gE[i] = isnothing(sol.resid) ? NaN : convert(Float64, sol.resid)
        gt[i] = t
        gp[i] = p
        # TODO: record algorithm details such as npts
    end

    @info "Started parameter sweep"
    t = time()
    sol = batchsolve(f, ps, T; callback=h5callback, kwargs...)
    t = time()-t
    @info @sprintf "Finished parameter sweep in %e (s) CPU time and %e (s) wall clock time" sum(read(gt)) t
    sol
end
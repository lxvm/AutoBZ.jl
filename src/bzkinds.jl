"""
    load_bz(::Val{:fbz}, seedname)

Load the full BZ using the utilities in `AutoBZCore`.
"""
function load_bz(::Val{:fbz}, seedname::String; atol=1e-5)
    A, B, = parse_wout(seedname * ".wout")
    FullBZ(A, B; atol=atol)
end

checkorthog(A::AbstractMatrix) = isdiag(transpose(A)*A)

sign_flip_tuples(n::Val{d}) where {d} = Iterators.product(ntuple(_ -> (1,-1), n)...)
sign_flip_matrices(n::Val{d}) where {d} = (Diagonal(SVector{d,Int}(A)) for A in sign_flip_tuples(n))
n_sign_flips(d::Integer) = 2^d


"""
    load_bz(::Val{:hbz}, seedname)

Load the half BZ, downfolded by inversion symmetries, using the utilities in
`AutoBZCore`. **Assumes orthogonal lattice vectors**
"""
function load_bz(::Val{:hbz}, seedname)
    A, B, = parse_wout(seedname * ".wout")
    d = checksquare(A)
    checkorthog(A) || @warn "Non-orthogonal lattice vectors detected with bzkind=:hbz. Unexpected behavior may occur"
    lims = CubicLimits(zeros(d), fill(0.5, d))
    syms = collect(sign_flip_matrices(Val(d)))
    SymmetricBZ(A, B, lims, syms)
end


function permutation_matrices(t::Val{n}) where {n}
    permutations = permutation_tuples(ntuple(identity, t))
    (StaticArrays.sacollect(SMatrix{n,n,Int,n^2}, ifelse(j == p[i], 1, 0) for i in 1:n, j in 1:n) for p in permutations)
end
permutation_tuples(C::NTuple{N,T}) where {N,T} = @inbounds((C[i], p...)::NTuple{N,T} for i in eachindex(C) for p in permutation_tuples(C[[j for j in eachindex(C) if j != i]]))
permutation_tuples(C::NTuple{1}) = C
n_permutations(n::Integer) = factorial(n)

"""
    cube_automorphisms(::Val{d}) where d

return a generator of the symmetries of the cube in `d` dimensions including the
identity.
"""
cube_automorphisms(n::Val{d}) where {d} = (S*P for S in sign_flip_matrices(n), P in permutation_matrices(n))
n_cube_automorphisms(d) = n_sign_flips(d) * n_permutations(d)

"""
    load_bz(::Val{:cubic_sym_ibz}, seedname)

Load the BZ, downfolded by cubic symmetries, using the utilities in
`AutoBZCore`. **Assumes orthogonal lattice vectors**
"""
function load_bz(::Val{:cubicsymibz}, seedname::String)
    A, B, = parse_wout(seedname * ".wout")
    d = checksquare(A)
    checkorthog(A) || @warn "Non-orthogonal lattice vectors detected with bzkind=:cubicsymibz. Unexpected behavior may occur"
    lims = TetrahedralLimits(fill(0.5, d))
    syms = vec(collect(cube_automorphisms(Val{d}())))
    SymmetricBZ(A, B, lims, syms)
end

export TetrahedralLimits, fermi_window_limits

"""
    TetrahedralLimits(a::SVector)
    TetrahedralLimits(a::CubicLimits)
    TetrahedralLimits(a, p)

A parametrization of the integration limits for a tetrahedron generated from the
automorphism group of the hypercube whose corners are `-a*p` and `a*p`. By
default, `p=0.5` which gives integration over the irreducible Brillouin zone of
the cube. If the entries of `a` vary then this implies the hypercube is
rectangular.
"""
struct TetrahedralLimits{d,T<:AbstractFloat} <: IntegrationLimits{d}
    a::SVector{d,T}
    p::T
end
TetrahedralLimits(a::SVector{d,T}) where {d,T<:AbstractFloat} = TetrahedralLimits(a, one(T)/2)
TetrahedralLimits(c::CubicLimits) = TetrahedralLimits(c.u-c.l)
(t::TetrahedralLimits)(x::Number) = TetrahedralLimits(pop(t.a), x/last(t.a))

box(t::TetrahedralLimits{d,T}) where {d,T} = StaticArrays.sacollect(SVector{d,Tuple{T,T}}, (zero(T), a) for a in t.a)
lower(t::TetrahedralLimits) = zero(t.p)
upper(t::TetrahedralLimits) = t.p*last(t.a)
nsyms(t::TetrahedralLimits) = n_cube_automorphisms(ndims(t))
symmetries(t::TetrahedralLimits) = cube_automorphisms(Val{ndims(t)}())

"""
    cube_automorphisms(d::Integer)

return a generator of the symmetries of the cube in `d` dimensions, optionally
including the identity.
"""
cube_automorphisms(n::Val{d}) where {d} = (S*P for S in sign_flip_matrices(n), P in permutation_matrices(n))
n_cube_automorphisms(d) = n_sign_flips(d) * n_permutations(d)

sign_flip_tuples(n::Val{d}) where {d} = Iterators.product(ntuple(_ -> (1,-1), n)...)
sign_flip_matrices(n::Val{d}) where {d} = (Diagonal(SVector{d,Int}(A)) for A in sign_flip_tuples(n))
n_sign_flips(d::Integer) = 2^d

permutation_matrices(t::Val{n}) where {n} = (StaticArrays.sacollect(SMatrix{n,n,Int,n^2}, ifelse(j == p[i], 1, 0) for i in 1:n, j in 1:n) for p in permutations(ntuple(identity, t)))
n_permutations(n::Integer) = factorial(n)
#= less performant code (at least when n=3)
permutation_matrices(t::Val{n}) where {n} = (SparseArrays.sparse(Base.OneTo(n), p, ones(n), n, n) for p in permutations(ntuple(identity, t)))
permutation_tuples(C::NTuple{N,T}) where {N,T} = @inbounds((C[i], p...)::NTuple{N,T} for i in eachindex(C) for p in permutation_tuples(C[[j for j in eachindex(C) if j != i]]))
permutation_tuples(C::NTuple{1}) = C;
=#

"""
    fermi_window_limits(Ω, β [; atol=0.0, rtol=1e-20, μ=0.0])

These limits are designed for integrating over the cubic FBZ first, then over ω
restricted to the interval where the Fermi window is larger than `atol`.
Choosing `atol` wisely is important to integrating the entire region of
interest, so i
"""
function fermi_window_limits(Ω, β; atol=0.0, rtol=1e-20, μ=0.0)
    Δω = fermi_window_halfwidth(Ω, β, select_fermi_atol(β*Ω, atol, rtol))
    CubicLimits(SVector(μ-Ω/2-Δω), SVector(μ-Ω/2+Δω))
end
select_fermi_atol(x, atol, rtol) = ifelse(x == zero(x), max(atol, 0.25rtol), max(atol, tanh(x/4)/x*rtol))
"""
    fermi_window_halfwidth(Ω, β, atol)
    fermi_window_halfwidth(β, atol)

One can show that β*Ω*fermi_window(ω, β, Ω) =
-tanh(β*Ω/2)/(cosh(β*(ω+Ω/2))/cosh(β*Ω/2)+1) >
-tanh(β*Ω/2)/(exp(abs(β*(ω+Ω/2)))/2cosh(β*Ω/2)+1)
as well as when Ω==0, β*fermi_window(ω, β, 0.0) = 
and these can be inverted to give a good bound on the width of the frequency
window for which the Fermi window function is greater than `atol`. Returns half
the width of this window.
"""
function fermi_window_halfwidth(Ω, β, atol)
    x = β*Ω
    if x == zero(x) || atol == zero(atol)
        fermi_window_halfwidth(β, atol)
    elseif tanh(x/4)/x > atol
        inv(β)*fermi_window_halfwidth_(x, atol)
    else
        error("choose `atol` under tanh(β*Ω/4)/(β*Ω), the maximum of the Fermi window")
    end
end
function fermi_window_halfwidth(β, atol)
    if β == zero(β) || atol == zero(atol)
        Inf
    elseif 1/4 > atol
        inv(β)*log(1/atol - 2)
    else
        error("choose `atol` under 1/4, the maximum of the Fermi window")
    end
end

fermi_window_halfwidth_(x, atol) = fermi_window_halfwidth_(float(x), atol)
function fermi_window_halfwidth_(x::AbstractFloat, atol)
    y = x/2
    log(2cosh(y)*(tanh(y)/(x*atol) - 1))
end
function fermi_window_halfwidth_(x::T, atol) where {T<:Union{Float32,Float64}}
    y = x/2
    abs_y = abs(y)
    y_large = Base.Math.H_LARGE_X(T)-1.0 # subtract 1 so 2cosh(x) won't overflow
    ifelse(abs_y > y_large, abs_y, log(2cosh(y))) + log(tanh(y)/(x*atol) - 1)
    # to be exact, add log1p(exp(-2abs_y)) to abs_y, but this is lost to roundoff
end
fermi_window_halfwidth_(x::Float16, atol) = Float16(fermi_window_halfwidth_(Float32(x), atol))
export TetrahedralLimits, fermi_window_limits

struct TetrahedralLimits1{d,N} <: IntegrationLimits{d}
    s::SVector{d,Float64}
    x::SVector{N,Float64}
end
TetrahedralLimits1(s) = TetrahedralLimits1(s, SVector{0,Float64}())
(l::TetrahedralLimits1)(x::Number) = TetrahedralLimits1(vcat(x, l.x), pop(l.s))

lower(::TetrahedralLimits1) = 0.0
upper(l::TetrahedralLimits1{d,0}) where {d} = 0.5last(l.s)
upper(l::TetrahedralLimits1) = first(l.x)#*last(l.s) need to double check the rescaling
nsym(::TetrahedralLimits1{d}) where {d} = 2^d * factorial(d)


"""
    TetrahedralLimits(d, s)

A parametrization of the integration limits for a tetrahedron generated from the
automorphism group of the cube on [0,s]^d
"""
struct TetrahedralLimits{d,N} <: IntegrationLimits{d}
    x::SVector{N,Float64}
    s::Float64
    TetrahedralLimits{d}(x::SVector{N}, s) where {d,N} = new{d,N}(x, s)
end
TetrahedralLimits(d::Int, s) = TetrahedralLimits{d}(SVector{0,Float64}(), s)
(t::TetrahedralLimits{d,N})(x::Number) where {d,N} = TetrahedralLimits{d-1}(vcat(x, t.x), t.s)

lower(::TetrahedralLimits) = 0.0
upper(t::TetrahedralLimits{d,0}) where {d} = 0.5t.s
upper(t::TetrahedralLimits) = first(t.x)
nsym(::TetrahedralLimits{d}) where {d} = 2^d * factorial(d)

symmetries(t::TetrahedralLimits) = cube_automorphisms(ndims(t))

"""
    cubic_symmetries(::Type{Val{d}}, [I=true])

return a generator of the symmetries of the cube in `d` dimensions, optionally
including the identity.
"""
cube_automorphisms(d::Integer) = (S*P for S in sign_flip_matrices(d), P in permutation_matrices(d))


sign_flip_tuples(d::Integer) = Iterators.product([(1,-1) for _ in 1:d]...)
sign_flip_matrices(d::Integer) = (Diagonal(SVector{d,Int}(A)) for A in sign_flip_tuples(d))

# More efficient algorithms (than recursion) for large n:
# Heap's algorithm
# Steinhaus–Johnson–Trotter algorithm
permutation_matrices(n::Integer) = (StaticArrays.sacollect(SMatrix{n,n,Int,n^2}, ifelse(j == p[i], 1, 0) for i in 1:n, j in 1:n) for p in permutation_tuples(Tuple(1:n)))
permutation_tuples(C::NTuple) = @inbounds((C[i], p...) for i in eachindex(C) for p in permutation_tuples(C[[j for j in eachindex(C) if j != i]]))
permutation_tuples(C::NTuple{1}) = C;

"""
    fermi_window_limits(Ω, β [; atol=0.0, rtol=1e-20, μ=0.0])

These limits are designed for integrating over the cubic FBZ first, then over ω
restricted to the interval where the Fermi window is larger than `atol`.
Choosing `atol` wisely is important to integrating the entire region of
interest, so i
"""
function fermi_window_limits(Ω, β; atol=0.0, rtol=1e-20, μ=0.0)
    Δω = fermi_window_halfwidth(Ω, β, select_fermi_atol(Ω, β, atol, rtol))
    CubicLimits(SVector(μ-Ω/2-Δω), SVector(μ-Ω/2+Δω))
end
select_fermi_atol(Ω, β, atol, rtol) = ifelse(Ω == zero(Ω) || β == zero(β), max(atol, 0.25rtol), max(atol, tanh(β*Ω/4)/(β*Ω)*rtol))
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
    if Ω == zero(Ω) || β == zero(β) || atol == zero(atol)
        fermi_window_halfwidth(β, atol)
    elseif tanh(β*Ω/4)/(β*Ω) > atol
        inv(β)*(log(2cosh(β*Ω/2)*(tanh(β*Ω/2)/(β*Ω*atol) - 1)))
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
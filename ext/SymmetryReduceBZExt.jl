module SymmetryReduceBZExt

using LinearAlgebra
using Polyhedra: Polyhedron, polyhedron, doubledescription, hrepiscomputed, hrep
using StaticArrays

using SymmetryReduceBZ
using AutoBZ: parse_wout, IBZ, DefaultPolyhedron, SymmetricBZ, CubicLimits, AbstractIteratedLimits, load_limits
import AutoBZ: load_bz, IteratedIntegration.fixandeliminate, IteratedIntegration.segments


include("ibzlims.jl")

function get_segs(vert::AbstractMatrix)
    rtol = atol = sqrt(eps(eltype(vert)))
    uniquepts=Vector{NTuple{2,eltype(vert)}}(undef, size(vert, 1))
    numpts = 0
    for i in axes(vert,1)
        v = vert[i,end]
        test = isapprox(v, atol=atol, rtol=rtol)
        if !any(x -> test(x[1]), @view(uniquepts[begin:begin+numpts-1,end]))
            numpts += 1
            uniquepts[numpts] = (v,v)
        end
    end
    @assert numpts >= 2 uniquepts
    resize!(uniquepts,numpts)
    sort!(uniquepts)
    top = pop!(uniquepts)
    for i in numpts-1:-1:1
        uniquepts[i] = top = (uniquepts[i][2],top[1])
    end
    return uniquepts
end

struct Polyhedron3{T<:Real} <: AbstractIteratedLimits{3,T}
    face_coord::Vector{Matrix{T}}
    segs::Vector{Tuple{T,T}}
end
function segments(ph::Polyhedron3, dim)
    @assert dim == 3
    return ph.segs
end

struct Polygon2{T<:Real} <: AbstractIteratedLimits{2,T}
    vert::Matrix{T}
    segs::Vector{Tuple{T,T}}
end
function segments(pg::Polygon2, dim)
    @assert dim == 2
    return pg.segs
end

function fixandeliminate(ph::Polyhedron3, z, ::Val{3})
    pg_vert = pg_vert_from_zslice(z, ph.face_coord)
    segs = get_segs(pg_vert)
    return Polygon2(pg_vert, segs)
end
function fixandeliminate(pg::Polygon2, y, ::Val{2})
    return CubicLimits(xlim_from_yslice(y, pg.vert)...)
end

function (::IBZ{Polyhedron})(a, real_latvecs, atom_types, atom_pos, coordinates; ibzformat="half-space", makeprim=false, convention="ordinary")
    hull_cart = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, ibzformat, makeprim, convention)
    hull = a' * polyhedron(doubledescription(hull_cart)) # rotate Cartesian basis to lattice basis in reciprocal coordinates
    hrepiscomputed(hull) || hrep(hull) # precompute hrep if it isn't already
    load_limits(hull)
end

function (::IBZ{DefaultPolyhedron})(a, real_latvecs, atom_types, atom_pos, coordinates; ibzformat="convex hull", makeprim=false, convention="ordinary", digits=12)
    hull = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, ibzformat, makeprim, convention)
    tri_idx = hull.simplices
    ph_vert = hull.points * a
    # tidy_vertices!(ph_vert, digits) # this should take care of rounding errors
    face_idx = faces_from_triangles(tri_idx, ph_vert)
    # face_idx = SymmetryReduceBZ.Utilities.get_uniquefacets(hull)
    face_coord = face_coord_from_idx(face_idx, ph_vert)
    segs = get_segs(ph_vert)
    Polyhedron3(face_coord, segs)
end

fixsign(x) = iszero(x) ? abs(x) : x
function tidy_vertices!(points, digits)
    for (i, p) in enumerate(points)
        points[i] = fixsign(round(p; digits=digits))
    end
    points
end

"""
    load_bz(::IBZ, seedname)

Use `SymmetryReduceBZ` to automatically load the IBZ. Since this method lives in
an extension module, make sure you write `using SymmetryReduceBZ` before `using
AutoBZ`.
"""
function load_bz(bz::IBZ, seedname::String; coordinates="lattice", rtol=nothing, atol=1e-9, digits=12)
    a, b, species, site, frac_lat, cart_lat = parse_wout(seedname * ".wout")
    real_latvecs = a    # we pick a, though we avoid using b because we would only get 6 digits of agreement from Wannier 90
    atom_species = unique(species)
    atom_types = map(e -> findfirst(==(e), atom_species) - 1, species)
    atom_pos = frac_lat
    # get symmetries
    sg = SymmetryReduceBZ.Symmetry.calc_spacegroup(real_latvecs, atom_types, atom_pos, coordinates)
    pg_ = SymmetryReduceBZ.Utilities.remove_duplicates(sg[2], rtol=something(rtol, sqrt(eps(float(maximum(real_latvecs))))), atol=atol)
    pg = convert(Vector{SMatrix{3,3,Float64,9}}, pg_) # deal with type instability in SymmetryReduceBZ
    syms = Ref(a') .* pg .* Ref(inv(a')) # rotate operator from Cartesian basis to lattice basis in reciprocal coordinates
    map!(s -> fixsign.(round.(s, digits=digits)), syms, syms)   # clean up matrix elements
    # get convex hull
    hull = bz(a, real_latvecs, atom_types, atom_pos, coordinates)
    # now limits and symmetries should be in reciprocal coordinates in the lattice basis
    SymmetricBZ(a, b, hull, syms)
end

end

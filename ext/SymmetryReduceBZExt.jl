module SymmetryReduceBZExt

using LinearAlgebra
using Polyhedra: polyhedron, doubledescription, hrepiscomputed, hrep
using StaticArrays

using SymmetryReduceBZ
using AutoBZ: parse_wout, IBZ, SymmetricBZ, CubicLimits, AbstractIteratedLimits, load_limits
import AutoBZ: load_bz, IteratedIntegration.fixandeliminate, IteratedIntegration.endpoints, IteratedIntegration.iterated_segs


include("ibzlims.jl")

struct Polyhedron{T<:Real} <: AbstractIteratedLimits{3,T}
    face_coord::Vector{Matrix{T}}
    endpoints::Tuple{T,T}
end
endpoints(ph::Polyhedron) = ph.endpoints

struct Polygon{T<:Real} <: AbstractIteratedLimits{2,T}
    vert::Matrix{T}
    endpoints::Tuple{T,T}
end
endpoints(pg::Polygon) = pg.endpoints

function fixandeliminate(ph::Polyhedron, z, digits=12)
    pg_vert = pg_vert_from_zslice(z, ph.face_coord)
    tidy_vertices!(pg_vert, digits)
    Polygon(pg_vert, get_lim(pg_vert))
end
function fixandeliminate(pg::Polygon, y)
    CubicLimits(xlim_from_yslice(y, pg.vert)...)
end

function iterated_segs(_, ph::Polyhedron, a, b, ::Val{initdivs}) where initdivs
    vert = unique(Iterators.flatten(@view(face[:,end]) for face in ph.face_coord))
    sort!(vert)
    tuple(vert...)
end
function iterated_segs(_, pg::Polygon, a, b, ::Val{initdivs}) where initdivs
    vert = unique(@view(pg.vert[:, end]))
    sort!(vert)
    tuple(vert...)
end

function load_polyhedra(a, real_latvecs, atom_types, atom_pos, coordinates; ibzformat="half-space", makeprim=false, convention="ordinary")
    hull_cart = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, ibzformat, makeprim, convention)
    hull = a' * polyhedron(doubledescription(hull_cart)) # rotate Cartesian basis to lattice basis in reciprocal coordinates
    hrepiscomputed(hull) || hrep(hull) # precompute hrep if it isn't already
    load_limits(hull)
end

function load_custom(a, real_latvecs, atom_types, atom_pos, coordinates; ibzformat="convex hull", makeprim=false, convention="ordinary", digits=12)
    hull = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, ibzformat, makeprim, convention)
    tri_idx = hull.simplices
    ph_vert = hull.points * a
    tidy_vertices!(ph_vert, digits) # this should take care of rounding errors
    face_idx = faces_from_triangles(tri_idx, ph_vert)
    # face_idx = SymmetryReduceBZ.Utilities.get_uniquefacets(hull)
    face_coord = face_coord_from_idx(face_idx, ph_vert)
    Polyhedron(face_coord, get_lim(ph_vert))
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
function load_bz(::IBZ, seedname::String, hull_func=load_custom; coordinates="lattice", rtol=nothing, atol=1e-9, digits=12)
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
    hull = hull_func(a, real_latvecs, atom_types, atom_pos, coordinates)
    # now limits and symmetries should be in reciprocal coordinates in the lattice basis
    SymmetricBZ(a, b, hull, syms)
end

end

module SymmetryReduceBZExt

using LinearAlgebra
using Polyhedra: polyhedron, doubledescription
using StaticArrays

if isdefined(Base, :get_extension)
    using SymmetryReduceBZ
    using AutoBZ: parse_wout, IBZ, SymmetricBZ, PolyhedralLimits, CubicLimits, AbstractIteratedLimits
    import AutoBZ: load_bz, IteratedIntegration.fixandeliminate, IteratedIntegration.endpoints
else
    using ..SymmetryReduceBZ
    using ..AutoBZ: parse_wout, IBZ, SymmetricBZ, PolyhedralLimits, CubicLimits, AbstractIteratedLimits
    import ..AutoBZ: load_bz, IteratedIntegration.fixandeliminate, IteratedIntegration.endpoints
end

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

function fixandeliminate(ph::Polyhedron, z)
    pg_vert = pg_vert_from_zslice(z, ph.face_coord)
    Polygon(pg_vert, get_lim(pg_vert))
end
function fixandeliminate(pg::Polygon, y)
    CubicLimits(xlim_from_yslice(y, pg.vert)...)
end

function load_polyhedra(a, real_latvecs, atom_types, atom_pos, coordinates; ibzformat="half-space", makeprim=false, convention="ordinary")
    hull_cart = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, ibzformat, makeprim, convention)
    hull = a' * polyhedron(doubledescription(hull_cart)) # rotate Cartesian basis to lattice basis in reciprocal coordinates
    PolyhedralLimits(hull)
end

function load_custom(a, real_latvecs, atom_types, atom_pos, coordinates; ibzformat="convex hull", makeprim=false, convention="ordinary")
    hull_cart = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, ibzformat, makeprim, convention)
    hull = hull_cart
    tri_idx = hull.simplices
    ph_vert = hull.points * a
    face_idx = faces_from_triangles(tri_idx, ph_vert)
    face_coord = face_coord_from_idx(face_idx, ph_vert)
    Polyhedron(face_coord, get_lim(ph_vert))
end

"""
    load_bz(::IBZ, seedname)

Use `SymmetryReduceBZ` to automatically load the IBZ. Since this method lives in
an extension module, make sure you write `using SymmetryReduceBZ` before `using
AutoBZ`.
"""
function load_bz(::IBZ, seedname::String, hull_func=load_custom; coordinates="lattice", rtol=nothing, atol=1e-9)
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
    # get convex hull
    hull = hull_func(a, real_latvecs, atom_types, atom_pos, coordinates)
    # now limits and symmetries should be in reciprocal coordinates in the lattice basis
    SymmetricBZ(a, b, hull, syms)
end

end

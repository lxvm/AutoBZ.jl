module SymmetryReduceBZExt

using LinearAlgebra
using Polyhedra: polyhedron, doubledescription
using StaticArrays

if isdefined(Base, :get_extension)
    using SymmetryReduceBZ
    using AutoBZ: parse_wout, IBZ, SymmetricBZ, PolyhedralLimits
    import AutoBZ: load_bz
else
    using ..SymmetryReduceBZ
    using ..AutoBZ: parse_wout, IBZ, SymmetricBZ, PolyhedralLimits
    import ..AutoBZ: load_bz
end

"""
    load_bz(::IBZ, seedname)

Use `SymmetryReduceBZ` to automatically load the IBZ. Since this method lives in
an extension module, make sure you write `using SymmetryReduceBZ` before `using
AutoBZ`.
"""
function load_bz(::IBZ, seedname::String; coordinates="lattice", ibzformat="half-space", makeprim=false, convention="ordinary", rtol=nothing, atol=1e-9)
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
    hull_cart = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, ibzformat, makeprim, convention)
    hull = a' * polyhedron(doubledescription(hull_cart)) # rotate Cartesian basis to lattice basis in reciprocal coordinates
    # now limits and symmetries should be in reciprocal coordinates in the lattice basis
    SymmetricBZ(a, b, PolyhedralLimits(hull), syms)
end

end
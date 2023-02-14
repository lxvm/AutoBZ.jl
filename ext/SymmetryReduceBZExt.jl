module SymmetryReduceBZExt

using SymmetryReduceBZ

using AutoBZ

function load_ibz(seedname::String; coordinates="lattice", ibzformat="half-space", makeprim=false, convention="angular", rtol=nothing, atol=1e-9)
    a, b, species, site, frac_lat, cart_lat = AutoBZ.parse_wout(seedname * ".wout")
    real_latvecs = reshape(reinterpret(Float64, a), 3, :)
    # SymmetryReduceBZ.Symmetry.calc_pointgroup(real_latvecs)
    atom_species = unique(species)
    atom_types = map(e -> findfirst(==(e), atom_species) - 1, species)
    atom_pos = reshape(reinterpret(Float64, frac_lat), 3, :)
    sg = SymmetryReduceBZ.Symmetry.calc_spacegroup(real_latvecs, atom_types, atom_pos, coordinates)
    pg = SymmetryReduceBZ.Utilities.remove_duplicates(sg[2], rtol=something(rtol, sqrt(eps(float(maximum(real_latvecs))))), atol=atol)
    # TODO: get the point group operations in lattice coordinates with A * pg * A'
    hs = calc_ibz(real_latvecs, atom_types, atom_pos, coordinates, ibzformat, makeprim, convention)
    SymmetricBZ(a, b, PolyhedronLimits(doubledescription(hs)), convert(Vector{SMatrix{3,3,Float64,9}}, pg))
end

end
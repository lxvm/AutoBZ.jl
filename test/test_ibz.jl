import SymmetryReduceBZ.Lattices: genlat_CUB, genlat_FCC, genlat_BCC,
  genlat_TET, genlat_BCT, genlat_ORC, genlat_ORCF, genlat_ORCI, genlat_ORCC,
  genlat_HEX, genlat_RHL, genlat_MCL, genlat_MCLC, genlat_TRI
import SymmetryReduceBZ.Symmetry: calc_bz, calc_ibz

using Test

"""
    ph_vol(n, tri_idx, ph_vert)

Estimate volume of polyhedron by equispaced integration.

# Arguments
- `n::Int64`: Number of integration points in z dimension
- `tri_idx::Matrix{Int32}`: nt x 3 array of indices of vertices of nt triangles
forming a triangulation of the polyhedron
- `ph_vert::Matrix{Float64}`: nv x 3 array of coordinates of nv polyhedron
vertices. `ph_vert[tri_idx[i,j],:]` gives the xyz coordinates of the jth vertex
of the ith triangle.

# Returns
- `vol::Float64`: Estimated volume of polyhedron
"""
function ph_vol(n::Int64, tri_idx::Matrix{Int32}, ph_vert::Matrix{Float64})

  # Vertices of faces of polyhedron, given by their indices
  face_idx = faces_from_triangles(tri_idx, ph_vert)

  # Vertices of faces of polyhedron, given by their ordered coordinates
  face_coord = face_coord_from_idx(face_idx, ph_vert)

  # Estimate volume of polyhedron by equispaced integration
  nz = n # Number of integration points in z dimension
  vol = 0.0
  zlims = get_lim(ph_vert) # z limits of integration
  dz = (zlims[2] - zlims[1]) / (nz + 1) # Integration step in z dimension
  for iz = 1:nz
    z = zlims[1] + iz * (zlims[2] - zlims[1]) / (nz + 1) # z coordinate

    # Vertices of polygon formed by intersection of polyhedron with z plane,
    # given by their ordered coordinates

    pg_vert = pg_vert_from_zslice(z, face_coord)

    ylims = get_lim(pg_vert) # y limits of integration

    # Number of integration points in y dimension is scaled by ratio of y and z interval lengths
    ny = round(nz * (ylims[2] - ylims[1]) / (zlims[2] - zlims[1]))
    dy = (ylims[2] - ylims[1]) / (ny + 1) # Integration step in y dimension

    for iy = 1:ny
      y = ylims[1] + iy * (ylims[2] - ylims[1]) / (ny + 1) # y coordinate

      xlims = xlim_from_yslice(y, pg_vert) # x limits of integration

      # Add volume of dy x dz rectangular prism of length x2-x1
      vol += (xlims[2] - xlims[1]) * dy * dz
    end
  end

  return vol

end

function test_vol(latvec::Matrix{Float64}, n::Int64)

  # Generate IBZ
  atom_types = [0]
  atom_pos = Array([0 0 0]')
  ibzformat = "convex hull"
  coordinates = "Cartesian"
  makeprim = false
  convention = "ordinary"
  ibz = calc_ibz(latvec, atom_types, atom_pos, coordinates, ibzformat,
    makeprim, convention)

  # Get triangles and vertices of IBZ
  tri_idx = ibz.simplices
  ph_vert = ibz.points

  vol = ph_vol(n, tri_idx, ph_vert) # Estimate volume of IBZ

  println("Estimated volume: ", vol)
  println("Actual volume: ", ibz.volume)

  return abs(vol - ibz.volume) / ibz.volume # Return relative error

end

function test_vol2(latvec::Matrix{Float64}, n::Int64)
    SymmetryReduceBZExt = Base.get_extension(AutoBZ, :SymmetryReduceBZExt)
    atom_types = [0]
    atom_pos = Array([0 0 0]')
    coordinates = "Cartesian"
    ibz_poly = SymmetryReduceBZExt.load_polyhedra(latvec, latvec, atom_types, atom_pos, coordinates)
    vol_poly, = nested_quadgk(ThunkIntegrand{3}(x -> 1.0), ibz_poly)
    ibz_hull = SymmetryReduceBZExt.load_custom(latvec, latvec, atom_types, atom_pos, coordinates)
    vol_hull, = nested_quadgk(ThunkIntegrand{3}(x -> 1.0), ibz_hull)

    # println("Reference volume: ", vol_poly)
    # println("Estimated volume: ", vol_hull)

    return abs(vol_poly - vol_hull) / vol_poly # Return relative error
end

@testset "IBZ volumes" begin

  a = 1.0         # Lattice constant
  b = 1.4         # Lattice constant
  c = 1.2         # Lattice constant
  alpha = pi / 6  # Lattice angle
  beta = pi / 3   # Lattice angle
  gamma = pi / 4  # Lattice angle
  n = 1000        # Number of integration points in z dimension
  tol = 1e-6      # Relative error tolerance

  # Estimate volumes of different lattices
  @test test_vol2(genlat_CUB(a), n) < tol
  @test test_vol2(genlat_FCC(a), n) < tol
  @test test_vol2(genlat_BCC(a), n) < tol
  @test test_vol2(genlat_TET(a, c), n) < tol
  @test test_vol2(genlat_BCT(a, c), n) < tol
  @test test_vol2(genlat_ORC(a, b, c), n) < tol
  @test test_vol2(genlat_ORCF(a, b, c), n) < tol
  @test test_vol2(genlat_ORCI(a, b, c), n) < tol
  @test test_vol2(genlat_ORCC(a, b, c), n) < tol
  @test test_vol2(genlat_HEX(a, c), n) < tol
  @test test_vol2(genlat_RHL(a, alpha), n) < tol
  @test test_vol2(genlat_MCL(a, b, c, alpha), n) < tol
  @test test_vol2(genlat_MCLC(a, b, c, alpha), n) < tol
  @test test_vol2(genlat_TRI(a, b, c, alpha, beta, gamma), n) < tol

end

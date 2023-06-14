import SymmetryReduceBZ.Lattices: genlat_CUB, genlat_FCC, genlat_BCC,
  genlat_TET, genlat_BCT, genlat_ORC, genlat_ORCF, genlat_ORCI, genlat_ORCC,
  genlat_HEX, genlat_RHL, genlat_MCL, genlat_MCLC, genlat_TRI
import SymmetryReduceBZ.Symmetry: calc_bz, calc_ibz
using Polyhedra: Polyhedron

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
  SymmetryReduceBZExt = Base.get_extension(AutoBZ, :SymmetryReduceBZExt)

  # Vertices of faces of polyhedron, given by their indices
  face_idx = SymmetryReduceBZExt.faces_from_triangles(tri_idx, ph_vert)

  # Vertices of faces of polyhedron, given by their ordered coordinates
  face_coord = SymmetryReduceBZExt.face_coord_from_idx(face_idx, ph_vert)

  # Estimate volume of polyhedron by equispaced integration
  nz = n # Number of integration points in z dimension
  vol = 0.0
  zlims = SymmetryReduceBZExt.get_lim(ph_vert) # z limits of integration
  dz = (zlims[2] - zlims[1]) / (nz + 1) # Integration step in z dimension
  for iz = 1:nz
    z = zlims[1] + iz * (zlims[2] - zlims[1]) / (nz + 1) # z coordinate

    # Vertices of polygon formed by intersection of polyhedron with z plane,
    # given by their ordered coordinates

    pg_vert = SymmetryReduceBZExt.pg_vert_from_zslice(z, face_coord)

    ylims = SymmetryReduceBZExt.get_lim(pg_vert) # y limits of integration

    # Number of integration points in y dimension is scaled by ratio of y and z interval lengths
    ny = round(nz * (ylims[2] - ylims[1]) / (zlims[2] - zlims[1]))
    dy = (ylims[2] - ylims[1]) / (ny + 1) # Integration step in y dimension

    for iy = 1:ny
      y = ylims[1] + iy * (ylims[2] - ylims[1]) / (ny + 1) # y coordinate

      xlims = SymmetryReduceBZExt.xlim_from_yslice(y, pg_vert) # x limits of integration

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

#   println("Estimated volume: ", vol)
#   println("Actual volume: ", ibz.volume)

  return abs(vol - ibz.volume) / ibz.volume # Return relative error

end

function test_vol2(latvec::Matrix{Float64}, n::Int64)
    atom_types = [0]
    atom_pos = Array([0 0 0]')
    ibzformat = "convex hull"
    coordinates = "Cartesian"
    makeprim = false
    convention = "ordinary"
    ibz = calc_ibz(latvec, atom_types, atom_pos, coordinates, ibzformat,
      makeprim, convention)
    ibz_hull = IBZ()(I, latvec, atom_types, atom_pos, coordinates)
    vol_hull, = nested_quad(x -> 1.0, ibz_hull)
    ibz_poly = IBZ{Polyhedron}()(I, latvec, atom_types, atom_pos, coordinates)
    vol_poly, = nested_quad(x -> 1.0, ibz_poly)

    # println("Reference volume: ", vol_poly)
    # println("Estimated volume: ", vol_hull)
    # println("Actual volume: ", ibz.volume)

    return max(abs(ibz.volume - vol_hull) / ibz.volume, abs(ibz.volume - vol_poly) / ibz.volume) # Return relative error
end

@testset "IBZ volumes" begin

  a = 1.0         # Lattice constant
  b = 1.4         # Lattice constant
  c = 1.2         # Lattice constant
  alpha = pi / 6  # Lattice angle
  beta = pi / 3   # Lattice angle
  gamma = pi / 4  # Lattice angle
  n = 1000        # Number of integration points in z dimension
  tol = 1e-2      # Relative error tolerance

  for routine in (test_vol, test_vol2)
    # Estimate volumes of different lattices
    @test routine(genlat_CUB(a), n) < tol
    @test routine(genlat_FCC(a), n) < tol
    @test routine(genlat_BCC(a), n) < tol
    @test routine(genlat_TET(a, c), n) < tol
    @test routine(genlat_BCT(a, c), n) < tol
    @test routine(genlat_ORC(a, b, c), n) < tol
    @test routine(genlat_ORCF(a, b, c), n) < tol
    @test routine(genlat_ORCI(a, b, c), n) < tol
    @test routine(genlat_ORCC(a, b, c), n) < tol
    @test routine(genlat_HEX(a, c), n) < tol
    @test routine(genlat_RHL(a, alpha), n) < tol
    @test routine(genlat_MCL(a, b, c, alpha), n) < tol
    @test routine(genlat_MCLC(a, b, c, alpha), n) < tol
    @test routine(genlat_TRI(a, b, c, alpha, beta, gamma), n) < tol
  end
end

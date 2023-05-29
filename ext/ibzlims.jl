import SymmetryReduceBZ.Utilities: sortpts_perm
import SymmetryReduceBZ.Utilities: sortpts2D

using LinearAlgebra

"""
    get_lim(vert)

Compute limits of integration for an integral over the final dimension of a
convex polytope of n vertices.

# Arguments
- `vert::Matrix{Float64}`: n x d array of vertices, where d is the final
dimension of the polytope, to be integrated over

# Returns
- `lim::Array{Float64, 1}`: 2-element array of limits of integration
"""
function get_lim(vert::Matrix{Float64})
  return extrema(vert[:, end])
end

"""
    faces_from_triangles(tri_idx, ph_vert)

Get faces of a polyhedron, represented by their indices, from its triangulation

# Arguments
- `tri_idx::Matrix{Int32}`: nt x 3 array of indices of vertices of nt triangles
forming a triangulation of the polyhedron
- `ph_vert::Matrix{Float64}`: nv x 3 array of coordinates of nv polyhedron
vertices. `ph_vert[tri_idx[i,j],:]` gives the xyz coordinates of the jth vertex
of the ith triangle.

# Returns
- `faces::Vector{Vector{Int32}}`: Vector of length nf of vectors of length nv_i
containing the unordered indices of the nv_i vertices contained in the ith face
of the polyhedron, where nf is the number of faces.

# Note
If you are obtaining the inputs to this function from a 3D Chull object, the
triangle indices tri_idx are given by the "simplices" attribute, and the vertex
coordinates ph_vert are given by the "points" attribute.
"""
function faces_from_triangles(tri_idx::Matrix{Int32}, ph_vert::Matrix{Float64})

  # Step 1: Get the normal vectors for each triangle. Note that the triangle
  # vertices are not necessarily given with respect to a consistent orientation,
  # so we can only get the normal vectors up to a sign.

  nvec = zeros(Float64, size(tri_idx))
  for i = 1:size(tri_idx, 1)
    # Get the coordinates of the vertices of the ith triangle
    tri = ph_vert[tri_idx[i, :], :]

    # Get the unit normal vector to the triangle
    u = cross(tri[2, :] - tri[1, :], tri[3, :] - tri[1, :])
    nvec[i, :] = u / norm(u)
  end

  # Step 2: Two triangles are in the same face if and only if (1) they share the
  # same normal vector, up to a sign, and (2) they are connected via a
  # continuous path through triangles which also have the same normal vector.
  # (1) alone is insufficient because a polygon can contain two parallel faces,
  # but (2) distinguishes this possibility. First, organize the triangles into
  # groups with the same normal vector up to a sign.

  # Get the unique unsigned normal vectors up to a tolerance
  isclose(x, y) = norm(x - y) < 1e-10 || norm(x + y) < 1e-10 # Define what it means for two vectors to be close
  nvec_unique = Vector{Vector{Float64}}() # Unique normal vectors
  for i = 1:size(nvec, 1) # Loop through all normal vectors
    # Check if ith normal vector is already in array of unique normal vectors
    u = nvec[i, :]
    nvec_is_unique = true # Initialize as true
    for j = 1:length(nvec_unique)
      if isclose(u, nvec_unique[j])
        nvec_is_unique = false
        break
      end
    end
    if nvec_is_unique
      push!(nvec_unique, u) # If not, add it
    end
  end

  # Divide triangles into groups of with shared unsigned normal vectors
  ngrp = size(nvec_unique, 1) # Number of groups
  grp_idx = Vector{Vector{Int32}}(undef, ngrp) # Indices of triangles in each group
  for i = 1:ngrp # Place each triangle into a group
    grp_idx[i] = Vector{Int32}() # Initialize as empty
    # Get indices of all triangles with ith unique normal vector
    for j = 1:size(nvec, 1)
      if isclose(nvec[j, :], nvec_unique[i])
        push!(grp_idx[i], j)
      end
    end
  end

  # Step 3: Next, split each group into subgroups of triangles which share at
  # least one vertex with another triangle in the group. Note that triangles in
  # parallel faces of a polyhedron will not share any vertices. Note also that
  # there can be at most two subgroups.

  face_idx = Vector{Vector{Int32}}() # Indices of vertices in each face
  for i = 1:ngrp # Loop through groups

    grpi_idx = grp_idx[i] # Indices of triangles in ith group
    # Place vertices of first triangle into subgroup 1 and remove from group
    subgrp1_vert_idx = tri_idx[grpi_idx[1], :] # Union of vertices of triangles in subgroup 1
    deleteat!(grpi_idx, 1)

    # Determine subgroup 1 by repeatedly looping through triangles in group
    placed_a_triangle = true # True if a triangle was placed in subgroup 1 in previous iteration
    while (placed_a_triangle) # Keep looping through triangles in group until all are placed
      placed_a_triangle = false # Initialize as false
      for j = 1:length(grpi_idx) # Loop through remaining triangles
        # If jth triangle has a vertex which appears in subgroup 1, add its
        # vertices to subgroup 1 and delete it from group
        if ((tri_idx[grpi_idx[j], 1]) in subgrp1_vert_idx) || ((tri_idx[grpi_idx[j], 2]) in subgrp1_vert_idx) || ((tri_idx[grpi_idx[j], 3]) in subgrp1_vert_idx)
          subgrp1_vert_idx = union(subgrp1_vert_idx, tri_idx[grpi_idx[j], :])
          deleteat!(grpi_idx, j)
          placed_a_triangle = true
          break
        end
      end
    end

    # Subgroup 1 is now complete and forms a face: add it to the list of faces
    push!(face_idx, subgrp1_vert_idx)

    # If there are any triangles left in the group, they are subgroup 2, and
    # also form a face; add to the list of faces
    if length(grpi_idx) > 0
      push!(face_idx, unique(tri_idx[grpi_idx, :]))
    end
  end

  return face_idx
end

"""
    face_coord_from_idx(face_idx, ph_vert)

Get coordinates of the vertices of the faces of a polyhedron from their indices.

# Arguments
- `face_idx::Vector{Vector{Int32}}`: Vector of length nf of vectors of length
nv_i containing the unordered indices of the nv_i vertices contained in the ith
face of the polyhedron, where nf is the number of faces.
- `ph_vert::Matrix{Float64}`: nv x 3 array of coordinates of the nv polyhedron
vertices. `ph_vert[face_idx[i][j],:]` gives the xyz coordinates of the jth
vertex of the ith face of the polyhedron.

# Returns
- `face_coord::Vector{Matrix{Float64}}`: Vector of length nf of matrices of size
nv_i x 3 containing the (clockwise or counter-clockwise) ordered coordinates
of the vertices of the ith face.
"""
function face_coord_from_idx(face_idx::Vector{Vector{Int32}}, ph_vert::Matrix{Float64})

  # Loop through face vertex index array and get the coordinates of the vertices
  # of each face
  face_coord = Matrix{Float64}[]
  for i in 1:length(face_idx)
    push!(face_coord, ph_vert[face_idx[i], :]) # Get coordinates of vertices
    p = sortpts_perm(face_coord[i]') # Sort vertices
    face_coord[i] = face_coord[i][p, :]
  end

  return face_coord
end

"""
    pg_vert_from_zslice(z, face_coord)

Get vertices of polygon formed by the intersection of a plane of constant z with
the faces of a polyhedron.

# Arguments
- `z::Float64`: z coordinate of plane
- `face_coord::Vector{Matrix{Float64}}`: Vector of length nf of matrices of size
nv_i x 3 containing the (clockwise or counter-clockwise) ordered coordinates of
the vertices of the ith face.

# Returns
- `pg_vert::Matrix{Float64}`: nv x 2 matrix of xy coordinates of the nv
(clockwise or counter-clockwise) ordered vertices of the polygon formed by the
intersection of the z plane with the polyhedron

# Note
Vertices which are shared between faces should be identical in floating point
arithmetic; that is, they should have come from a common array listing the
unique vertices of the polyhedron.

z must be between the minimum and maximum z coordinates of the polyhedron, and
not equal to one of them.
"""
function pg_vert_from_zslice(z::Float64, face_coord::Vector{Matrix{Float64}})

  pg_vert = Vector{Vector{Float64}}() # Matrix of vertices of the polygon
  for i = 1:length(face_coord) # Loop through faces
    face = face_coord[i]

    # Loop through ordered pairs of vertices in the face, and check whether the
    # line segment connecting them intersects the z plane.
    nvi = size(face, 1)
    for j = 1:nvi
      jp1 = mod1(j + 1, nvi)
      z1 = face[j, 3]
      z2 = face[jp1, 3]
      if (z1 <= z && z2 >= z) || (z1 >= z && z2 <= z)
        # Find the point of intersection and add it to the list of polygon
        # vertices
        t = (z - z1) / (z2 - z1)
        # dot syntax removes some allocations/array copies as do views
        v = @. t * @view(face[jp1, 1:2]) + (1 - t) * @view(face[j, 1:2])
        push!(pg_vert, v)
      end
    end
  end

  pg_vert1 = stack(pg_vert)'  # new variable name since type may change
  # pg_vert1 = hcat(pg_vert...)'  # not type stable

  # There will be redundant vertices in the verts array because of shared
  # edges between faces; these should be exactly equal (in floating point
  # arithmetic) because the vertices in each face should come from a common
  # list of unique vertices of the polyhedron. Remove the redundant vertices.
  pg_vert2 = unique(pg_vert1, dims=1)

  # Sort the points in the polygon by their angle with respect to the centroid
  p = sortpts2D(pg_vert2') # permutation which sorts the points
  pg_vert3 = pg_vert2[p, :]

  return pg_vert3
end

"""
    xlim_from_yslice(y, pg_vert)

Get x coordinates of the intersection between a line of constant y and the
boundary of a polygon.

# Arguments
- `y::Float64`: y coordinate of line
- `pg_vert::Matrix{Float64}`: nv x 2 matrix of xy coordinates of the nv
(clockwise or counter-clockwise) ordered vertices of the polygon

# Returns
- `xlims::Vector{Float64}`: pair of x coordinates of intersection between the
line and the polygon boundary

# Note
y must be between the minimum and maximum y coordinates of the polygon, and not
equal to one of them.
"""
function xlim_from_yslice(y::Float64, pg_vert::Matrix{Float64})

  # Loop through ordered pairs of vertices, and check whether the line segment
  # connecting them intersects the line of constant y
  lb = NaN  # undefined Float64
  k = 0
  nv = size(pg_vert, 1)
  for j = 1:nv
    jp1 = mod1(j + 1, nv)
    y1 = pg_vert[j, 2]
    y2 = pg_vert[jp1, 2]

    # If y1 and y2 are on opposite sides of y, then line intersects the edge. If
    # y = y1, then line intersects a vertex and we include it. If y = y2, then
    # line intersects a vertex, but we don't include it to avoid double-counting.
    if (y1 < y && y2 > y) || (y1 > y && y2 < y) || y1 == y
      # Find the point of intersection and add it to the list of intersection points
      t = (y - y1) / (y2 - y1)
      k += 1
      lim = t * pg_vert[jp1, 1] + (1 - t) * pg_vert[j, 1]
      if k == 1
        lb = lim
      elseif (k == 2) # Found two unique intersection points
        return extrema((lb, lim))
      end
    end
  end

  # If we reach the end and have only found one intersection point, then the
  # line is tangent to a single vertex, and both x limits are the same
  @assert k == 1 "could not find intersection with polygon"
  return (lb, lb)
end

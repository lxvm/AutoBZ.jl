# Band diagrams

Plotting band diagrams on k-paths is a common visualization tool and we show
some examples of how to do so within `AutoBZ`. We use the
[Brillouin.jl](https://github.com/thchr/Brillouin.jl) package to select and
visualize k-paths.

## Energy bands

In the following example, we plot the band structure of graphene defined as a tight binding model in PythTB.
This serves to reproduce the [PythTB homepage example](https://www.physics.rutgers.edu/pythtb/) using AutoBZ.jl and highlight the [`load_pythtb_data`](@ref) interface.

```@example
using PyCall
py"""
from pythtb import *

# lattice vectors and orbital positions
lat=[[1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]]
orb=[[1./3., 1./3.], [2./3., 2./3.]]

# two-dimensional tight-binding model
gra=tb_model(2, 2, lat, orb)

# define hopping between orbitals
gra.set_hop(-1.0, 0, 1, [ 0, 0])
gra.set_hop(-1.0, 1, 0, [ 1, 0])
gra.set_hop(-1.0, 1, 0, [ 0, 1])

k=[[0.0, 0.0],[1./3., 2./3.],[0.5,0.5]]
path=gra.k_path(k, 100)
"""
gra = py"gra"
(k_vec,k_dist,k_node)=py"path"

using AutoBZ
h,bz = load_pythtb_data(gra)

using LinearAlgebra
hk = getproperty.(eigen.(h.(eachrow(k_vec))), :values)

using GLMakie
fig = Figure()
ax = Axis(fig[1,1]; xticks = (k_node, ["Γ", "K", "M"]))
series!(ax, k_dist, stack(hk))
save("grapheneband.png", fig); nothing # hide
```

![Graphene band kpath](grapheneband.png)

## Band velocities

!!! note "In progress"
    This will be added soon

To overlay band velocities on the k-path
module BrillouinPlotlyJSExt

using LinearAlgebra: checksquare, dot, norm

using AutoBZ: Hamiltonian, AbstractHamiltonianInterp, gauge, Lattice, AbstractVelocityInterp, coord
using Brillouin
using PlotlyJS
import PlotlyJS: plot

function plot(kpi::KPathInterpolant, h::AbstractHamiltonianInterp, args...; kwargs...)
    gauge(h) isa Hamiltonian || throw(ArgumentError("Please provide Hamiltonian in the Hamiltonian gauge"))
    data = h.(kpi)
    nband = checksquare(eltype(parent(h).c))
    bands = [ map(h_k -> h_k.values[i], data) for i in 1:nband ]
    return plot(kpi, bands, args...; kwargs...)
end

function plot(kpi::KPathInterpolant, hv::AbstractVelocityInterp, args...; kwargs...)
    gauge(hv) isa Hamiltonian || throw(ArgumentError("Please provide Hamiltonian in the Hamiltonian gauge"))
    coord(hv) isa Lattice || throw(ArgumentError("Please provide velocities in lattice coordinates"))
    data = hv.(kpi)
    nband = length(data[begin][begin].values)
    bands = [ map(h_k -> h_k[1].values[i], data) for i in 1:nband ]

    local_xs         = cumdists.(cartesianize(kpi).kpaths)

    # plot bands and k-lines/labels
    quivers = Vector{Vector{GenericTrace{Dict{Symbol,Any}}}}()
    start_idx = 1
    for (path_idx, (local_x, labels)) in enumerate(zip(local_xs, kpi.labels))
        stop_idx = start_idx+length(local_x)-1
        push!(quivers, [quiver(
                [(m == 1 ? local_x[n-start_idx+1] : data[n][1].values[i]) for m in 1:2, n in start_idx:(stop_idx-1)],
                [(m == 1 ? local_x[n-start_idx+2]-local_x[n-start_idx+1] : dot(real(getindex.(data[n][2],i,i)), kpi[n+1]-kpi[n])) for m in 1:2, n in start_idx:(stop_idx-1)],
                scale=1.0, xaxis="x$path_idx", yaxis="y", marker_color=colors.tab20[i]) for i in 1:nband]
        )
        start_idx = stop_idx + 1
    end
    plt = plot(kpi, bands, args...; kwargs...)
    for (j,q) in enumerate(quivers)
        # TODO: color each band and distinguish the subplots
        addtraces!(plt, q...)
    end
    # plot(quivers[1])
    return plt
end

# taken from https://github.com/JuliaPlots/PlotlyJS.jl/issues/230
"""
    quiver(x, u; scale = 1.0)

Returns a trace, to be plotted using PlotlyJS, corresponding to a quiver plot.

Inputs:

      - x::Matrix{Float64}: matrix of size 2 x n, origin of each arrow
      - u::Matrix{Float64}: matrix of size 2 x n, components of each arrow. The arrow endpoint is at (x + scale*u, y + scale*v)

We assume that the (x,y) coordinates are given in each column of x and u. The number of
columns is the number of points to plot.

Kwargs:

      - scale::Float64 magnification factor for the arrow lengths

Example:

    using LinearAlgebra # This is needed by quiver
    using PlotlyJS      # Plotting is done using PlotlyJS
    x = [0. 1. 2.;
         1. 2. 3.]
    u = [10. 20. 30;
          0. 10. 50.]
    plot([quiver(x, u, scale = .1)], Layout(yaxis_scaleanchor="x"))
    # Layout(...) is important to make sure that x and y axis have the same scale
"""
function quiver(x::Matrix{Float64},u::Matrix{Float64}; scale::Float64 = 1.0, kws...)
    n = size(x,2)
    s = 8
    X = Matrix{Float64}(undef,2,s*n)
    for i=1:n
        k = s*(i-1)

        r = x[:,i]
        V = scale * u[:,i]

        dist = norm(V)
        arrow_h = 0.1dist     # height of arrowhead
        arrow_w = 0.5arrow_h  # halfwidth of arrowhead
        U1 = V ./ dist        # vector of arrowhead height
        U2 = [-U1[2], U1[1]]  # vector of arrowhead halfwidth
        U1 *= arrow_h
        U2 *= arrow_w

        X[:,k+1] = r
        r += V
        X[:,k+2:k+s] = [r-U1 [NaN, NaN] r r-U1+U2 r-U1-U2 r [NaN, NaN]]
    end
    trace = scatter(fill="toself"; kws...)
    trace[:x] = X[1,:]; trace[:y] = X[2,:]
    return trace
end

end

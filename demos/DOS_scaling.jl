#=
In this script, we extract timings for adaptive and equispace OC for diminishing
eta
=#

using HDF5
using Plots

using AutoBZ

# define the periods of the axes of the Brillouin zone for example material
period = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
H = AutoBZ.Applications.load_hamiltonian("svo_hr.dat"; period=period)

# Define problem parameters
ω = 0.0 # eV
etas = 2.0 .^ (-1:-1:-4) # eV
μ = 12.3958 # eV (SrVO3)

# initialize integration limits
c = AutoBZ.CubicLimits(H.period)
const t = AutoBZ.Applications.TetrahedralLimits(c)

# set error tolerances
atol = 1e-3
rtol = 0.0

# allocate results
aints = Vector{Float64}(undef, length(etas))
aerrs = Vector{Float64}(undef, length(etas))
atimes = Vector{Float64}(undef, length(etas))

eints = Vector{Float64}(undef, length(etas))
eerrs = Vector{Float64}(undef, length(etas))
etimes = Vector{Float64}(undef, length(etas))

# save equispace precomputations
equi_save = (npt1=0, pre1=nothing, npt2=0, pre2=nothing)

for (i, eta) in enumerate(etas)
    # construct integrand for given eta
    Σ = AutoBZ.Applications.EtaEnergy(eta)
    D = AutoBZ.Applications.DOSIntegrand(H, ω, Σ, μ)
    # time adaptive code
    r = @timed AutoBZ.iterated_integration(D, t; callback=AutoBZ.Applications.contract, atol=atol, rtol=rtol)
    aints[i], aerrs[i] = r.value
    atimes[i] = r.time
    @show eta atimes[i] aints[i] aerrs[i]
    println()
    # time equispace code
    r = @timed AutoBZ.automatic_equispace_integration(D, t; atol=atol, rtol=rtol, pre_eval=AutoBZ.Applications.pre_eval_contract, equi_save...)
    global eints[i], eerrs[i], equi_save = r.value
    etimes[i] = r.time
    @show eta etimes[i] eints[i] eerrs[i]
    println()
end

h5open("OC_results_scaling.h5", "w") do h5
    g = create_group(h5, "adaptive")
    g["time"] = atimes
    g["int"] = aints
    g["err"] = aerrs
    h = create_group(h5, "equispace")
    h["time"] = etimes
    h["int"] = eints
    h["err"] = eerrs
end

plt = plot(; scale=:log10, xguide="η (eV)", yguide="Wall time (s)", title="DOS scaling, D(ω=$ω),atol=$atol")
plot!(plt, etas[2:end], atimes[2:end]; color=1, markershape=:x, label="adaptive")
plot!(plt, etas[2:end], x -> 1e-2*log(1/x)^3; color=1, ls=:dash, label="O(log(1/η)³)")
plot!(plt, etas[2:end], etimes[2:end]; color=2, markershape=:x, label="equispace")
plot!(plt, etas[2:end], x -> 1e-3/x^3; color=2, ls=:dash, label="O(1/η³)")
savefig("DOS_scaling.png")

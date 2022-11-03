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
omegas = range(-1, 1; length=50) # eV
etas = 2.0 .^ (-1:-1:-5) # eV
μ = 12.3958 # eV (SrVO3)

# initialize integration limits
c = AutoBZ.CubicLimits(H.period)
t = AutoBZ.Applications.TetrahedralLimits(c)

# set error tolerances
atol = 1e-3
rtol = 0.0

# allocate results
aints = Matrix{Float64}(undef, length(omegas), length(etas))
aerrs = Matrix{Float64}(undef, length(omegas), length(etas))
atimes = Matrix{Float64}(undef, length(omegas), length(etas))

eints = Matrix{Float64}(undef, length(omegas), length(etas))
eerrs = Matrix{Float64}(undef, length(omegas), length(etas))
etimes = Matrix{Float64}(undef, length(omegas), length(etas))

# save equispace precomputations
equi_save = (npt1=0, pre1=nothing, npt2=0, pre2=nothing)

for (j, eta) in enumerate(etas)
    # construct integrand for given eta
    Σ = AutoBZ.Applications.EtaEnergy(eta)
    for (i, omega) in enumerate(omegas)
        D = AutoBZ.Applications.DOSIntegrand(H, omega, Σ, μ)
        # time adaptive code
        r = @timed AutoBZ.iterated_integration(D, t; callback=AutoBZ.Applications.contract, atol=atol, rtol=rtol)
        aints[i,j], aerrs[i,j] = r.value
        atimes[i,j] = r.time
        # @show eta omega atimes[i,j] aints[i,j] aerrs[i,j]
        # println()
        # time equispace code
        r = @timed AutoBZ.automatic_equispace_integration(D, t; atol=atol, rtol=rtol, pre_eval=AutoBZ.Applications.pre_eval_contract, equi_save...)
        global eints[i,j], eerrs[i,j], equi_save = r.value
        etimes[i,j] = r.time
        # @show eta omega etimes[i,j] eints[i,j] eerrs[i,j]
        # println()
    end
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
plot!(plt, etas[2:end], vec(sum(atimes[:,2:end], dims=1)); color=1, markershape=:x, label="adaptive")
plot!(plt, etas[2:end], x -> 5e-1*log(1/x)^3; color=1, ls=:dash, label="O(log(1/η)³)")
plot!(plt, etas[2:end], vec(sum(etimes[:,2:end], dims=1)); color=2, markershape=:x, label="equispace")
plot!(plt, etas[2:end], x -> 1e-4/x^3; color=2, ls=:dash, label="O(1/η³)")
savefig("DOS_scaling.png")

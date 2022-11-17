#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using Plots

using AutoBZ
using AutoBZ.Applications

# define the periods of the axes of the Brillouin zone for example material
b = round(2π/3.858560, digits=6)
# Load the Wannier Hamiltonian as a Fourier series
HV_full  = load_hamiltonian_velocities("svo_hr.dat", "svo_r.dat"; period=b)
HV_inter = load_hamiltonian_velocities("svo_hr.dat", "svo_r.dat"; period=b, kind=:inter)
HV_intra = load_hamiltonian_velocities("svo_hr.dat", "svo_r.dat"; period=b, kind=:intra)

# Define problem parameters
Ωs = range(0, 2, length=200) # eV
η = 1.0 # eV
μ = 12.3958 # eV
β = inv(sqrt(η*8.617333262e-5*0.5*300/pi)) # eV # Fermi liquid scaling

# initialize integrand and limits
Σ = EtaEnergy(η)
c = CubicLimits(period(HV_full))
t = TetrahedralLimits(c)

# set error tolerances
atol = 1e-3
rtol = 1e-3
npt = 50

# pre-evaluate the series on the k-grid
t_ = time()
pre_full = pre_eval_contract(HV_full, t, npt)
@info "pre-evaluated Wannier Hamiltonian in $(time()-t_) seconds"

t_ = time()
pre_inter = pre_eval_contract(HV_inter, t, npt)
@info "pre-evaluated inter-band Hamiltonian in $(time()-t_) seconds"

t_ = time()
pre_intra = pre_eval_contract(HV_intra, t, npt)
@info "pre-evaluated intra-band Hamiltonian in $(time()-t_) seconds"

σ_full  = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
σ_inter = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
σ_intra = Vector{eltype(OCIntegrand)}(undef, length(Ωs))
for (i, Ω) in enumerate(Ωs)
    σ = OCIntegrand(HV_full, Σ, Ω, β, μ)
    Eσ_full  = EquispaceOCIntegrand(σ, t, npt, pre_full)
    Eσ_inter = EquispaceOCIntegrand(σ, t, npt, pre_inter)
    Eσ_intra = EquispaceOCIntegrand(σ, t, npt, pre_intra)

    f = fermi_window_limits(Ω, β)
    σ_full[i],  = iterated_integration(Eσ_full,  f; atol=atol, rtol=rtol)
    σ_inter[i], = iterated_integration(Eσ_inter, f; atol=atol, rtol=rtol)
    σ_intra[i], = iterated_integration(Eσ_intra, f; atol=atol, rtol=rtol)
end

plt = plot(; xguide="Ω (eV)", yguide="σ", title="SrVO3 OC, $npt kpts/dim, η=$η eV, β=$(round(β,digits=2)) ev⁻¹")
plot!(plt, Ωs, map(real∘first, σ_full); label="full")
plot!(plt, Ωs, map(real∘first, σ_inter); label="inter")
plot!(plt, Ωs, map(real∘first, σ_intra); label="intra")
plot!(plt, Ωs, map(real∘first∘+, σ_intra, σ_inter); label="sum")
savefig(plt, "OC_bands.png")
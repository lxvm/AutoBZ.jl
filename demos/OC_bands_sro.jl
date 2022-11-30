#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using Plots

using AutoBZ
using AutoBZ.Applications

# define the periods of the axes of the Brillouin zone for example material
b = 1.0
# Load the Wannier Hamiltonian as a Fourier series
HV_full  = load_hamiltonian_velocities("sro_hr.dat", "sro_r.dat"; period=b)
HV_inter = load_hamiltonian_velocities("sro_hr.dat", "sro_r.dat"; period=b, kind=:inter)
HV_intra = load_hamiltonian_velocities("sro_hr.dat", "sro_r.dat"; period=b, kind=:intra)

# Define problem parameters
Ωs = range(0, 4, length=200) # eV
η = 0.002 # eV
μ = 11.366595 # eV
β = 200.0 # 1/eV

# initialize integrand and limits
Σ = EtaEnergy(η)
c = CubicLimits(period(HV_full))
t = TetrahedralLimits(c)

# set error tolerances
atol = 1e-7
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

t_full = t_inter = t_intra = 0.0


σ_full  = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
σ_inter = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
σ_intra = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
for (i, Ω) in enumerate(Ωs)
    σ = KineticIntegrand(HV_full, Σ, β, μ, 0, Ω)
    Eσ_full  = EquispaceKineticIntegrand(σ, t, npt, pre_full)
    Eσ_inter = EquispaceKineticIntegrand(σ, t, npt, pre_inter)
    Eσ_intra = EquispaceKineticIntegrand(σ, t, npt, pre_intra)

    f = fermi_window_limits(Ω, β)
    
    global t_ = time()
    σ_full[i],  = iterated_integration(Eσ_full,  f; atol=atol, rtol=rtol, order=7)
    global t_full += time() - t_

    global t_ = time()
    σ_inter[i], = iterated_integration(Eσ_inter, f; atol=atol, rtol=rtol, order=7)
    global t_inter += time() - t_
    
    global t_ = time()
    σ_intra[i], = iterated_integration(Eσ_intra, f; atol=atol, rtol=rtol, order=7)
    global t_intra += time() - t_
end

@info "Integrated :full  at $(length(Ωs)) points in $t_full s"
@info "Integrated :inter at $(length(Ωs)) points in $t_inter s"
@info "Integrated :intra at $(length(Ωs)) points in $t_intra s"

plt = plot(; yscale=:log10, xguide="Ω (eV)", yguide="σ (units ?)", title="SRO OC, $npt kpts/dim, η=$η eV, β=$(round(β,digits=2)) ev⁻¹")
plot!(plt, Ωs, map(real∘first, σ_full); label="full")
plot!(plt, Ωs, map(real∘first, σ_inter); label="inter")
plot!(plt, Ωs, map(real∘first, σ_intra); label="intra")
plot!(plt, Ωs, map(real∘first∘+, σ_intra, σ_inter); label="sum")
savefig(plt, "OC_bands_sro.png")


err_plt = plot(; ylim=(1e-16, 1), yscale=:log10, xguide="Ω (eV)", yguide="|σ_full - σ_sum|", title="SRO OC, $npt kpts/dim, η=$η eV, β=$(round(β,digits=2)) ev⁻¹")
plot!(err_plt, Ωs, map((x,y,z) -> abs(first(x - y - z)), σ_full, σ_intra, σ_inter); label="")
plot!(err_plt, Ωs, x -> atol; ls=:dash, color=:red, label="atol")
savefig(err_plt, "OC_bands_sro_error.png")
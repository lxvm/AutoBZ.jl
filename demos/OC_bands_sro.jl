#=
In this script we compute OC at single point using the interface in AutoBZ.jl
=#

using Plots

using AutoBZ

seedname="sro"
# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
HV_orbit, FBZ_orbit = load_wannier90_data(seedname; velocity_kind=:orbital)
HV_inter, FBZ_inter = load_wannier90_data(seedname; velocity_kind=:interband)
HV_intra, FBZ_intra = load_wannier90_data(seedname; velocity_kind=:intraband)

ibz_limits = AutoBZ.TetrahedralLimits(period(HV_orbit)) # Cubic symmetries
IBZ = IrreducibleBZ(FBZ_orbit.a, FBZ_orbit.b, ibz_limits)


# Define problem parameters
Ωs = range(0, 4, length=200) # eV
η = 0.002 # eV
μ = 11.366595 # eV
β = 200.0 # 1/eV

# initialize self energy
Σ = EtaSelfEnergy(η)
shift!(HV_orbit, μ)
shift!(HV_inter, μ)
shift!(HV_intra, μ)

# set error tolerances
atol = 1e-7
rtol = 1e-3
npt = 50

# pre-evaluate the series on the k-grid
t_ = time()
pre_orbit = pre_eval_contract(HV_orbit, IBZ, npt)
@info "pre-evaluated Wannier Hamiltonian in $(time()-t_) seconds"

t_ = time()
pre_inter = pre_eval_contract(HV_inter, IBZ, npt)
@info "pre-evaluated inter-band Hamiltonian in $(time()-t_) seconds"

t_ = time()
pre_intra = pre_eval_contract(HV_intra, IBZ, npt)
@info "pre-evaluated intra-band Hamiltonian in $(time()-t_) seconds"

t_orbit = t_inter = t_intra = 0.0


σ_orbit = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
σ_inter = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
σ_intra = Vector{eltype(KineticIntegrand)}(undef, length(Ωs))
for (i, Ω) in enumerate(Ωs)
    σ = KineticIntegrand(HV_orbit, Σ, β, 0, Ω)
    Eσ_orbit = EquispaceKineticIntegrand(σ, IBZ, npt, pre_orbit)
    Eσ_inter = EquispaceKineticIntegrand(σ, IBZ, npt, pre_inter)
    Eσ_intra = EquispaceKineticIntegrand(σ, IBZ, npt, pre_intra)

    f = fermi_window_limits(Ω, β)
    
    global t_ = time()
    σ_orbit[i], = AutoBZ.iterated_integration(Eσ_orbit, f; atol=atol, rtol=rtol, order=7)
    global t_orbit += time() - t_

    global t_ = time()
    σ_inter[i], = AutoBZ.iterated_integration(Eσ_inter, f; atol=atol, rtol=rtol, order=7)
    global t_inter += time() - t_
    
    global t_ = time()
    σ_intra[i], = AutoBZ.iterated_integration(Eσ_intra, f; atol=atol, rtol=rtol, order=7)
    global t_intra += time() - t_
end

@info "Integrated :orbital   at $(length(Ωs)) points in $t_orbit s"
@info "Integrated :interband at $(length(Ωs)) points in $t_inter s"
@info "Integrated :intraband at $(length(Ωs)) points in $t_intra s"

plt = plot(; yscale=:log10, xguide="Ω (eV)", yguide="σ (units ?)", title="SRO OC, $npt kpts/dim, η=$η eV, β=$(round(β,digits=2)) ev⁻¹")
plot!(plt, Ωs, map(real∘first, σ_orbit); label="orbit")
plot!(plt, Ωs, map(real∘first, σ_inter); label="inter")
plot!(plt, Ωs, map(real∘first, σ_intra); label="intra")
plot!(plt, Ωs, map(real∘first∘+, σ_intra, σ_inter); label="sum")
savefig(plt, "OC_bands_sro.png")


err_plt = plot(; ylim=(1e-16, 1), yscale=:log10, xguide="Ω (eV)", yguide="|σ_orbit - σ_sum|", title="SRO OC, $npt kpts/dim, η=$η eV, β=$(round(β,digits=2)) ev⁻¹")
plot!(err_plt, Ωs, map((x,y,z) -> abs(first(x - y - z)), σ_orbit, σ_intra, σ_inter); label="")
plot!(err_plt, Ωs, x -> atol; ls=:dash, color=:red, label="atol")
savefig(err_plt, "OC_bands_sro_error.png")
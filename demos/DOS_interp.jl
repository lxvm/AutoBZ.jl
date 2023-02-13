#=
In this script we interpolate DOS over a frequency interval using the interface in AutoBZ.jl
=#

using Plots

using AutoBZ.Jobs
using AutoBZ.Jobs.AdaptChebInterp

# Load the Wannier Hamiltonian as a Fourier series and the Brillouin zone 
H, FBZ = load_wannier90_data("svo")

IBZ = Jobs.cubic_sym_ibz(FBZ; atol=1e-5) # for lattices with cubic symmetry only


# Define problem parameters
ω_lo = -2.0 # eV
ω_hi =  2.0 # eV
η = 0.1 # eV
μ = 12.3958 # eV

shift!(H, μ) # shift the Fermi energy to zero
Σ = EtaSelfEnergy(η)

# set error tolerances
atol = 1e-3
rtol = 0.0

# set interpolation parameters
interp_atol=1e-3
order = 4
fast_order = 15

# D = first ∘ DOSIntegrator(IBZ, H, Σ; atol=atol, rtol=rtol) # adaptive default
# D = first ∘ DOSIntegrator(IBZ, H, Σ; routine=Jobs.AutoBZ.equispace_integration)
D = first ∘ DOSIntegrator(IBZ, H, Σ; atol=atol, rtol=rtol, routine=Jobs.AutoBZ.automatic_equispace_integration)

adaptchebinterp(D, ω_lo, ω_hi; atol=1.0, order=order)
t_ = time()
p1 = adaptchebinterp(D, ω_lo, ω_hi; atol=interp_atol, order=order)
@info "rigorous interpolation took $(time()-t_) s"


fastadaptchebinterp(D, ω_lo, ω_hi; atol=1.0, order=fast_order)
t_ = time()
p2 = fastadaptchebinterp(D, ω_lo, ω_hi; atol=interp_atol, order=fast_order)
@info "fast interpolation took $(time()-t_) s"

nodes = Float64[]
for panel in p1.searchtree
    iszero(panel.val) && continue
    push!(nodes, AdaptChebInterp.chebpoints(order, panel.a, panel.b)...)
end
# unique!(nodes)

fast_nodes = Float64[]
for panel in p2.searchtree
    iszero(panel.val) && continue
    push!(fast_nodes, AdaptChebInterp.chebpoints(fast_order, panel.a, panel.b)...)
end
# unique!(fast_nodes)

plot(range(ω_lo, ω_hi, length=1000), p1; xguide="ω (eV)", yguide="DOS(ω)", label="rigorous")
scatter!(nodes, fill(-0.2, length(nodes)); color=1, markerstrokewidth=0, markershape=:diamond, label="", alpha=0.25)
plot!(range(ω_lo, ω_hi, length=1000), p2; color=2, label="fast")
scatter!(fast_nodes, fill(-0.6, length(order+1)); color=2, markerstrokewidth=0, markershape=:diamond, label="", alpha=0.25)
savefig("DOS_interp.png")

plot(range(ω_lo, ω_hi, length=1000), x -> abs(p1(x)-p2(x)); xguide="ω (eV)", color=:black, yguide="DOS(ω) interpolant difference", yscale=:log10, title="Interpolation atol $interp_atol", ylim=(1e-5, 1), label="|fast-rigorous|", legend=:bottomright)
scatter!(nodes, fill(0.8, length(nodes)); color=1, markerstrokewidth=0, markershape=:diamond, alpha=0.25, label="rigorous nodes")
scatter!(fast_nodes, fill(0.5, length(fast_nodes)); color=2, markerstrokewidth=0, markershape=:diamond, alpha=0.25, label="fast nodes")
savefig("DOS_interp_error.png")

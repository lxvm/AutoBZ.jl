using LinearAlgebra

using StaticArrays
using OffsetArrays
using Plots

using AutoBZ

a₀ = 1.0 # lattice constant of hexagonal lattice
a = sqrt(3)a₀ # lattice constant of triangular lattice
b = 2*pi/a
C = OffsetArray(zeros(SMatrix{2,2,ComplexF64,4}, (5,5)), -2:2, -2:2)
C[1,1]   = C[1,-2] = C[-2,1] = [0 1; 0 0]
C[-1,-1] = C[-1,2] = C[2,-1] = [0 0; 1 0]
H = FourierSeries(C, b)

M = SMatrix{2,2,Float64,4}([1/2 1/2sqrt(3); 1/2 -1/2sqrt(3)])
iM = inv(M)

ks = range(-b/2, b/2, length=513)
Hs = map(k -> H(M*SVector{2}(k[1], k[2])), Iterators.product(ks, ks))
band_plt = plot(; xguide="kx", yguide="ky", zguide="H(kx,ky)", title="Graphene band structure")
surface!(band_plt, ks, ks, map(e -> real(eigvals(Matrix(e))[1]), Hs))
surface!(band_plt, ks, ks, map(e -> real(eigvals(Matrix(e))[2]), Hs))
savefig(band_plt, "graphene_bands.png")

T = 100.0 # K
kB = 8.617333262e-5 # eV/K

lambda(x, T, kB) = -x*fermi′(inv(kB*T)*x)/(kB*T^2)
function evaluate_integrand(f, T, kB)
    ξ_k, ξ_q = real.(det.(f))
    if ξ_k == ξ_q
        # when the integrand is ill defined, return its limiting value ∂λ/∂ξ
        return lambda(ξ_k, T, kB)*(2*inv(kB*T)*fermi′(ξ_k*inv(kB*T))/fermi(ξ_k*inv(kB*T)) - inv(ξ_k) - inv(kB*T))
    else
        return (lambda(ξ_k, T, kB) - lambda(ξ_q, T, kB))/(ξ_k-ξ_q)
    end
end
ints = map(k -> evaluate_integrand(ManyOffsetsFourierSeries(H, M*SVector(1.0, 2.0))(k), T, kB), map(k -> M * SVector(k), Iterators.product(ks, ks)))
int_plt = heatmap(ks, ks, map(abs, ints); xguide="kx", yguide="ky", title="|Integrand| with q=(1,2)", color=:BuGn)
# plot(ks, map(k -> evaluate_integrand(ManyOffsetsFourierSeries(H, M*SVector(1.0, 2.0))(SVector(0.0, k)), T, kB), ks))
savefig(int_plt, "graphene_integrand.png")

FBZ = AutoBZ.CubicLimits(period(H))

# set error tolerances
atol = 1e-3
rtol = 0.0

r = Matrix{Float64}(undef, length(ks), length(ks))
for (i, kx) in enumerate(ks), (j, ky) in enumerate(ks)
    q = M * SVector((kx, ky))
    f = ManyOffsetsFourierSeries(H, q)
    integrand = WannierIntegrand(evaluate_integrand, f, (T, kB))
    r[i,j], = AutoBZ.iterated_integration(integrand, FBZ; atol=atol, rtol=rtol)
end
plt = heatmap(ks, ks, map(abs, r); xguide="qx", yguide="qy", title="|Integral| in q-space", color=:BuGn)
savefig(plt, "graphene_integral.png")

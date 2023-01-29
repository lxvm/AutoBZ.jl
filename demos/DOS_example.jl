using StaticArrays
using OffsetArrays
using Plots

using AutoBZ.Jobs

ns = 1:3
plt = plot(; xguide="ω", yguide="DOS")

nω = 81
ints = Vector{Float64}(undef, nω)

for (j, n) in enumerate(ns)
    a = 2pi; b = 2pi/a
    C = OffsetArray(zeros(SMatrix{1,1,ComplexF64,1}, ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k == i ? j : 0, n))] = SMatrix{1,1,ComplexF64,1}(1/2n)
    end
    H = Jobs.FourierSeries(C, b)

    ωs = range(-sqrt(2), sqrt(2), length=nω)
    ηs = (0.2, 0.1, 0.05)
    
    # construct IBZ integration limits for the cubic symmetries
    A = a*one(SMatrix{n,n,Float64,n^2})
    FBZ = Jobs.FullBZ(A)
    IBZ = Jobs.cubic_sym_ibz(FBZ)
    
    # set error tolerances
    atol = 1e-4
    rtol = 0.0
    
    for (k,η) in enumerate(ηs)
        Threads.@threads for (i, ω) in collect(enumerate(ωs))
            # D = SafeDOSIntegrand(H, EtaSelfEnergy(η), ω)
            D = DOSIntegrand(H, EtaSelfEnergy(η), ω)
            ints[i], = Jobs.iterated_integration(D, IBZ; atol=atol, rtol=rtol)
        end
        plot!(plt, ωs, ints; label=k == length(ηs) ? "n=$n" : "", color=j, alpha=k/length(ηs))#(1/(1+η^3)))
    end
end
savefig("tb_dos.png")

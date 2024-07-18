using AutoBZ, HChebInterp, LinearAlgebra

seed = "svo" # Wannier90 output files

# model parameters
T = 64.0 # K
kB = 8.617333262e-5 # eV/K
β = inv(kB*T)

T₀ = 300.0 # K
Z = 0.5
η = (kB*pi/T₀/Z)*T^2 # eV
Σ = EtaSelfEnergy(η)

# DOS calculation

bzalg = IAI()
abstol = 1e-5
reltol = 1e-4

ω = 2.0 # eV
μ = 12.5 # eV

h, bz = load_wannier90_data(seed;
    interp=HamiltonianInterp,
    gauge=Wannier(),
    bz=CubicSymIBZ(),
)

trg_solver = TrGlocSolver(Σ, h, bz, bzalg; ω, μ, abstol, reltol)

trg_batch = let μ=μ, N=Threads.nthreads(), trg_chnl = Channel{typeof(trg_solver)}(N)
    for _ in 1:N
        put!(trg_chnl, TrGlocSolver(Σ, h, bz, bzalg; ω, μ, abstol, reltol))
    end
    BatchFunction() do ωs
        out = Vector{ComplexF64}(undef, length(ωs))
        Threads.@threads for n in 1:N
            _solver = take!(trg_chnl)
            for i in n:N:length(out)
                AutoBZ.update_trgloc!(_solver; ω = ωs[i], μ)
                sol = solve!(_solver)
                out[i] = sol.value
            end
            put!(trg_chnl, _solver)
        end
        return out
    end
end
atol = abstol*100
trg_interp = hchebinterp(trg_batch, -ω, ω; atol)

dos_x = range(-ω, ω; length=1000)
dos_dat = -imag(trg_interp.(dos_x))/pi./det(bz.B)

# electron density calculation

bzalg = IAI()
falg = QuadGKJL()
abstol = 1e-5
reltol = 1e-4

μ = 12.5 # eV
μmax = 2.0 # eV

h, bz = load_wannier90_data(seed;
    interp=HamiltonianInterp,
    gauge=Hamiltonian(),
    bz=CubicSymIBZ(),
)

ρ_solver = ElectronDensitySolver(h, bz, bzalg, Σ, falg; β, μ, abstol, reltol)

ρ_batch = let β=β, N=Threads.nthreads(), ρ_chnl = Channel{typeof(ρ_solver)}(N)
    for _ in 1:N
        put!(ρ_chnl, ElectronDensitySolver(h, bz, bzalg, Σ, falg; β, μ, abstol, reltol))
    end
    BatchFunction() do μs
        out = Vector{Float64}(undef, length(μs))
        Threads.@threads for n in 1:N
            _solver = take!(ρ_chnl)
            for i in n:N:length(out)
                AutoBZ.update_density!(_solver; β, μ=μs[i])
                sol = solve!(_solver)
                out[i] = sol.value
            end
            put!(ρ_chnl, _solver)
        end
        return out
    end
end
atol = 1e-2
ρ_interp = hchebinterp(ρ_batch, μ-μmax, μ+μmax; atol)

ρ_x = range(μ-μmax, μ+μmax; length=1000)
ρ_dat = ρ_interp.(ρ_x)./det(bz.B)

# Optical conductivity calculation

bzalg = IAI()
falg = QuadGKJL()
abstol = AutoBZ.AutoBZCore.IteratedIntegration.AuxValue(1e-5, abs(det(bz.B))*1e-2)
reltol = 1e-4

μ = 12.5 # eV
Ω = 0.3 # eV

hv, bz = load_wannier90_data(seed;
    interp=CovariantVelocityInterp,
    gauge=Hamiltonian(),
    coord=Cartesian(),
    vcomp=Whole(),
    bz=CubicSymIBZ(),
)

a = AutoBZ.freq2rad(η/AutoBZ.velocity_bound(AutoBZ.parentseries(hv)))
s = AutoBZ.AutoBZCore.IteratedIntegration.AuxValue(inv(a*nsyms(bz)*abs(det(bz.B))), 0.1)
σ_solver = AuxOpticalConductivitySolver(hv, bz, bzalg, Σ, falg; β, μ, Ω, abstol, reltol, scale_inner=s)

σ_batch = let μ=μ, β=β, s=s, N=Threads.nthreads(), σ_chnl = Channel{typeof(σ_solver)}(N)
    for _ in 1:N
        put!(σ_chnl, AuxOpticalConductivitySolver(hv, bz, bzalg, Σ, falg; β, μ, Ω, abstol, reltol, scale_inner=s))
    end
    BatchFunction() do Ωs
        out = Vector{AutoBZ.SMatrix{3,3,ComplexF64,9}}(undef, length(Ωs))
        Threads.@threads for n in 1:N
            _solver = take!(σ_chnl)
            for i in n:N:length(out)
                AutoBZ.update_auxoc!(_solver; β, μ, Ω=Ωs[i])
                sol = solve!(_solver)
                out[i] = sol.value.val
            end
            put!(σ_chnl, _solver)
        end
        return out
    end
end
atol = 1e-2
rtol = 1e-3
σ_interp = hchebinterp(σ_batch, zero(Ω), Ω; atol, rtol)

σ_x = range(zero(Ω), Ω; length=1000)
σ_dat = real.(getindex.(σ_interp.(σ_x), 1, 1)) * 0.61657354879193 # conversion from 1/Ang to kS/cm


begin
    using CairoMakie
    set_theme!(;
        fontsize=40,
        linewidth=8,
        Axis = (;
            xgridvisible=false,
            xtickalign = 1,
            ygridvisible=false,
            ytickalign = 1,
            yticklabelrotation=pi/2,
            spinewidth=3,
            xtickwidth=3,
            xticksize=10,
            ytickwidth=3,
            yticksize=10,
        ),
    )
    fig = Figure(size=(1800, 600))
    dos_axis = Axis(fig[1,1];
        xlabel="ω (eV)",
        ylabel="DOS (1/eV)",
        limits=((-2, 2), (0, 6))
    )
    lines!(dos_axis, dos_x, dos_dat)
    text!(dos_axis, 0.85*4-2, 0.85*6-0; text="(a)")
    ρ_axis = Axis(fig[1,2];
        xlabel="μ (eV)",
        ylabel="n",
        limits=((10.5, 14.5), (0, 3)),
        xticks=LinearTicks(3),
    )
    lines!(ρ_axis, ρ_x, ρ_dat)
    text!(ρ_axis, (1-0.9)*4+10.5, 0.85*3+0; text="(b)")
    σ_axis = Axis(fig[1,3];
        xlabel="Ω (eV)",
        ylabel="σ (kS/cm)",
        limits=((0.0, 0.3), (0, 15))
    )
    lines!(σ_axis, σ_x, σ_dat)
    text!(σ_axis, 0.85*0.3+0, 0.85*15-0; text="(c)")

    save("figure.png", fig)
end

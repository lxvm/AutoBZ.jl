using AutoBZ, HChebInterp, LinearAlgebra

seed = "svo" # Wannier90 output files

# model parameters
T = 24.0 # K
kB = 8.617333262e-5 # eV/K
β = inv(kB*T)

T₀ = 300.0 # K
Z = 0.5
η = (kB*pi/T₀/Z)*T^2 # eV
Σ = EtaSelfEnergy(η)

kalgs = [PTR(npt=100), PTR(npt=300), IAI()]
kalg_labels = ["Nₖ=100", "Nₖ=300", "IAI"]
colors = [:forestgreen, :dodgerblue, :orangered]


# DOS calculation

abstol = 1e-5
reltol = 1e-4

atol = 1e-3

ω = 2.0 # eV
μ = 12.5 # eV

h, bz = load_wannier90_data(seed;
    interp=HamiltonianInterp,
    gauge=Wannier(),
    bz=CubicSymIBZ(),
)

trg_interps = map(kalgs) do bzalg
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
    hchebinterp(trg_batch, -ω, ω; atol)
end

#=
# electron density calculation

falg = QuadGKJL()
abstol = 1e-5
reltol = 1e-4

atol = 1e-3

μ = 12.5 # eV
μmax = 2.0 # eV

h, bz = load_wannier90_data(seed;
    interp=HamiltonianInterp,
    gauge=Hamiltonian(),
    bz=CubicSymIBZ(),
)

ρ_interps = map(kalgs) do bzalg
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
    hchebinterp(ρ_batch, μ-μmax, μ+μmax; atol)
end
=#


# Optical conductivity calculation

falg = QuadGKJL()
abstol = AutoBZ.AutoBZCore.IteratedIntegration.AuxValue(1e-5, abs(det(bz.B))*1e-2)
reltol = 1e-4

atol = 1e-2
rtol = 1e-3

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
s = inv(nsyms(bz)*abs(det(bz.B)))*AutoBZ.AutoBZCore.IteratedIntegration.AuxValue(inv(a), 0.1)
σ_interps = map(kalgs) do bzalg
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
    hchebinterp(σ_batch, zero(Ω), Ω; atol, rtol)
end


# trg_interps = [trg_interp_iai, trg_interp_300]
dos_x = range(-ω, ω, length=10_000)
# ρ_interps = [ρ_interp_iai, ρ_interp_300]
# ρ_x = range(μ-μmax, μ+μmax; length=10_000)
# σ_interps = [σ_interp_iai, σ_interp_300]
σ_x = range(zero(Ω), Ω; length=10_000)
begin
    using CairoMakie
    set_theme!(;
        fontsize=80,
        linewidth=8,
        Axis = (;
            xgridvisible=false,
            xtickalign = 1,
            ygridvisible=false,
            ytickalign = 1,
            yticklabelrotation=pi/2,
            spinewidth=6,
            xtickwidth=6,
            xticksize=20,
            ytickwidth=6,
            yticksize=20,
        ),
        Series = (;
            color = colors,
            linewidth=4,
        ),
        Legend = (;
            framewidth=6,
            patchsize=(80, 6),
        ),
    )
    fig = Figure(size=(3200, 1200))
    dos_axis = Axis(fig[1,1];
        xlabel="ω (eV)",
        ylabel="DOS (1/eV)",
        limits=((-1.2, 1.49), (0, 7))
    )
    dos_dat = [(dos_x, -imag.(trg.(dos_x))./pi./det(bz.B)) for trg in trg_interps]
    series!(dos_axis, dos_dat)
    #=
    dos_axis_inset = Axis(fig;
        bbox=BBox(175, 375, 300, 500),
        xticks=WilkinsonTicks(3),
        yticks=WilkinsonTicks(3),
        limits=((0.45, 0.5), (1.3, 2.0)),
        xticklabelsize=20,
        yticklabelsize=20,
        spinewidth=2,
        xtickwidth=2,
        xticksize=8,
        ytickwidth=2,
        yticksize=8,
    )
    series!(dos_axis_inset, dos_dat)
    =#
    text!(dos_axis, (1-0.95)*2.69-1.2, 0.85*7-0; text="(a)")
    #=
    ρ_axis = Axis(fig[1,2];
        xlabel="μ (eV)",
        ylabel="n",
        limits=((10.5, 14.5), (0, 3)),
        xticks=LinearTicks(3),
    )
    ρ_dat = [(ρ_x, ρ.(ρ_x)./det(bz.B)) for ρ in ρ_interps]
    series!(ρ_axis, ρ_dat)
    ρ_axis_inset = Axis(fig;
        bbox=BBox(775, 975, 300, 500),
        xticks=LinearTicks(3),
        yticks=LinearTicks(3),
        limits=((13.7, 13.9), (2.9, 3.0)),
        xticklabelsize=20,
        yticklabelsize=20,
        spinewidth=2,
        xtickwidth=2,
        xticksize=8,
        ytickwidth=2,
        yticksize=8,
    )
    series!(ρ_axis_inset, ρ_dat)
    text!(ρ_axis, (1-0.9)*4+10.5, 0.85*3+0; text="(b)")
    =#
    σ_axis = Axis(fig[1,2];
        xlabel="Ω (eV)",
        ylabel="σ (kS/cm)",
        limits=((0.0, 0.3), (0, 11))
    )
    σ_dat = [(σ_x, real.(getindex.(σ.(σ_x), 1, 1)) .* 0.61657354879193) for σ in σ_interps]
    series!(σ_axis, σ_dat)
    #=
    σ_axis_inset = Axis(fig;
        bbox=BBox(900, 1100, 300, 500),
        xticks=LinearTicks(3),
        yticks=LinearTicks(3),
        limits=((0.15, 0.17), (1.0, 2.0)),
        xticklabelsize=20,
        yticklabelsize=20,
        spinewidth=2,
        xtickwidth=2,
        xticksize=8,
        ytickwidth=2,
        yticksize=8,
    )
    series!(σ_axis_inset, σ_dat)
    =#
    text!(σ_axis, 0.85*0.3+0, 0.85*11-0; text="(b)")
    Legend(fig[1,3], [LineElement(; color) for color in colors], kalg_labels)

    save("figure.png", fig)
end

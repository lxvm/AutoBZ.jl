using Plots
using ProgressBars

using AutoBZ
using AutoBZ.Applications

function load_dataset(filename)
    data = readlines(filename)
    Omega = Vector{Float64}(undef, length(data))
    sigma = Vector{Float64}(undef, length(data))
    for (i, line) in enumerate(data)
        Omega[i], sigma[i] = map(x -> parse(Float64, x), split(line))
    end
    return (Omega=Omega, sigma=sigma)
end

function generate_dataset_adaptive(; atol=1e-3, rtol=1e-5)
    b = round(2π/3.858560, digits=6) # 1/angstroms
    HVnoA = Applications.load_hamiltonian_velocities("svo_hr.dat"; period=b)
    HVwA = Applications.load_hamiltonian_velocities("svo_hr.dat", "svo_r.dat"; period=b)

    mu = 12.3958 # eV
    
    segbufs = AutoBZ.alloc_segbufs(Float64, eltype(KineticIntegrand), Float64, 4)

    for (eta, beta, filename) in [
        (0.1, 1.0, "optic_c_eta.1_beta1_new.dat"),
        (0.01, 100.0, "optic_c_eta.01_beta100.dat"),
        (0.002, 200.0, "optic_c_eta.002_beta200.dat"),
    ]
        Sigma = Applications.EtaEnergy(eta)
        data = load_dataset(filename)
        sigma = Vector{Float64}(undef, length(data.Omega))

        for (HV, label) in [(HVnoA, ""), (HVwA, "_berry")]
            c = CubicLimits(period(HV))
            t = TetrahedralLimits(c)

            for (i, Om) in ProgressBar(enumerate(data.Omega))
                σ = KineticIntegrand(HV, Sigma, beta, mu, 0, Om)
                f = fermi_window_limits(Om, beta)
                l = CompositeLimits(t, f)
                int, err = iterated_integration(σ, l; atol=max(atol, rtol*abs(data.sigma[1])), segbufs=segbufs)
                sigma[i] = 616.57354879193*real(int[1,1])
            end

            open("oc_validate_adaptive_eta$(string(eta)[2:end])_beta$(Int(beta))$(label).dat", "w") do f
                for (Om, s) in zip(data.Omega, sigma)
                    write(f, string(Om, " ", s, "\n"))
                end
            end
        end
    end
end

function generate_dataset_equispace(; atol=1e-5, rtol=0.0)
    mu = 12.3958 # eV
    b = round(2π/3.858560, digits=6) # 1/angstroms
    npt = 40
    
    HVnoA = Applications.load_hamiltonian_velocities("svo_hr.dat"; period=b)
    cnoA = CubicLimits(period(HVnoA))
    tnoA = TetrahedralLimits(cnoA)
    prenoA = pre_eval_contract(HVnoA, tnoA, npt)
    
    HVwA = Applications.load_hamiltonian_velocities("svo_hr.dat", "svo_r.dat"; period=b)
    cwA = CubicLimits(period(HVwA))
    twA = TetrahedralLimits(cwA)
    prewA = pre_eval_contract(HVwA, twA, npt)
    
    for (eta, beta, filename) in [
        (0.1, 1.0, "optic_c_eta.1_beta1_new.dat"),
        (0.01, 100.0, "optic_c_eta.01_beta100.dat"),
        (0.002, 200.0, "optic_c_eta.002_beta200.dat"),
    ]
        Sigma = Applications.EtaEnergy(eta)
        data = load_dataset(filename)
        sigma = Vector{Float64}(undef, length(data.Omega))

        for (HV, pre, label) in [(HVnoA, prenoA, ""), (HVwA, prewA, "_berry")]

            for (i, Om) in ProgressBar(enumerate(data.Omega))
                σ = KineticIntegrand(HV, Sigma, beta, mu, 0, Om)
                f = fermi_window_limits(Om, beta)
                Eσ = EquispaceKineticIntegrand(σ, t, npt, pre)
                int, err = iterated_integration(Eσ, f; atol=atol, rtol=rtol)
                sigma[i] = 616.57354879193*real(int[1,1])
            end

            open("oc_validate_equispace_eta$(string(eta)[2:end])_beta$(Int(beta))$(label).dat", "w") do f
                for (Om, s) in zip(data.Omega, sigma)
                    write(f, string(Om, " ", s, "\n"))
                end
            end
        end
    end
end

function vis_dataset()
    for (eta, beta, filename) in [
        (0.1, 1.0, "optic_c_eta.1_beta1_new.dat"),
        (0.01, 100.0, "optic_c_eta.01_beta100.dat"),
        (0.002, 200.0, "optic_c_eta.002_beta200.dat"),
    ]
        validata = load_dataset(filename)
        inds = (validata.Omega .< 2) .& (validata.sigma .> 0)
        
        plt = plot(; xguide="Ω (eV)", yguide="σ₁₁ (Ω⁻¹ cm⁻¹)", yscale=:log10, title="η=$eta meV, β=$beta 1/eV")
        plot!(plt, validata.Omega[inds], validata.sigma[inds]; label="TRIQS", markershape=:x)
        err_plt = plot(; xguide="Ω (eV)", yguide="relative error (%)", yscale=:log10, ylim=(1e-3,100), title="η=$eta meV, β=$beta 1/eV")
    
        for label in ("", "_berry")
            testdata_adaptive = load_dataset("oc_validate_adaptive_eta$(string(eta)[2:end])_beta$(Int(beta))$(label).dat")
            testdata_equispace = load_dataset("oc_validate_equispace_eta$(string(eta)[2:end])_beta$(Int(beta))$(label).dat")
            

            plot!(plt, testdata_adaptive.Omega[inds], testdata_adaptive.sigma[inds]; label="AutoBZ_adaptive$label", markershape=:x)
            plot!(plt, testdata_equispace.Omega[inds], testdata_equispace.sigma[inds]; label="AutoBZ_equispace$label", markershape=:x)

            plot!(err_plt, validata.Omega, 100abs.(testdata_adaptive.sigma .- validata.sigma) ./ abs.(validata.sigma); label="adaptive$label")
            plot!(err_plt, validata.Omega, 100abs.(testdata_equispace.sigma .- validata.sigma) ./ abs.(validata.sigma); label="equispace$label")
        end
        savefig(plt, "OC_validation_result_eta$(string(eta)[2:end])_beta$(Int(beta)).png")
        savefig(err_plt, "OC_validation_error_eta$(string(eta)[2:end])_beta$(Int(beta)).png")
    end
end
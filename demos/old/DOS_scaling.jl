#=
In this script, we extract timings for adaptive and equispace OC for diminishing
eta
=#

using LinearAlgebra
using ProgressBars
using HDF5
using Plots

using AutoBZ
using AutoBZ.Jobs

function initialize_data(seedname="svo", μ=12.3958)
# Load the Wannier Hamiltonian as a Fourier series
H, FBZ = load_wannier90_data(seedname; compact=:S)
shift!(HV, μ)

IBZ = Jobs.cubic_sym_ibz(FBZ; atol=1e-5) # for lattices with cubic symmetry only

# Define problem parameters
omegas = collect(range(-1, 1; length=3))#00)) # eV
etas = collect(2.0 .^ (-1:-1:-10)) # eV

# set error tolerances (the most generous is always chosen)
atol_ = 3 # decimal places
atol = 10.0^(-atol_)
rtol_ = Inf # significant digits
rtol = 10.0^(-rtol_)

# allocate results
ints = Matrix{Float64}(undef, length(omegas), length(etas))
errs = Matrix{Float64}(undef, length(omegas), length(etas))
times = Matrix{Float64}(undef, length(omegas), length(etas))

H, omegas, etas, IBZ, "atol-$(atol_)_rtol-$(rtol_)", atol, rtol, ints, errs, times
end

function write_h5(filename, etas, omegas, ints, errs, times)
    h5open(filename, "w") do h5
        h5["eta"] = etas
        h5["omega"] = omegas
        h5["int"] = ints
        h5["err"] = errs
        h5["time"] = times
    end
end

function read_h5(filename)
    h5open(filename, "r") do h5
        etas = read_dataset(h5, "eta")
        omegas = read_dataset(h5, "omega")
        ints = read_dataset(h5, "int")
        errs = read_dataset(h5, "err")
        times = read_dataset(h5, "time")
        return etas, omegas, ints, errs, times
    end
end


function write_h5_npt(filename, etas, omegas, npt1, npt2)
    h5open(filename, "w") do h5
        h5["eta"] = etas
        h5["omega"] = omegas
        h5["npt1"] = npt1
        h5["npt2"] = npt2
    end
end

function read_h5_npt(filename)
    h5open(filename, "r") do h5
        etas = read_dataset(h5, "eta")
        omegas = read_dataset(h5, "omega")
        npt1 = read_dataset(h5, "npt1")
        npt2 = read_dataset(h5, "npt2")
        return etas, omegas, npt1, npt2
    end
end


function equispace_scaling()
    H, omegas, etas, t, id, atol, rtol, ints, errs, times = initialize_data()
    equi_save = (npt1=0, pre1=nothing, npt2=0, pre2=nothing)
    npt1 = Matrix{Int64}(undef, length(omegas), length(etas))
    npt2 = Matrix{Int64}(undef, length(omegas), length(etas))
    for (j, eta) in enumerate(etas)
        @info "starting log2(eta)=$(log2(eta))"
        Σ = EtaSelfEnergy(eta)
        for (i, omega) in ProgressBar(enumerate(omegas))
            D = DOSIntegrand(H, Σ, omega)
            r = @timed AutoBZ.automatic_equispace_integration(D, t; atol=atol, rtol=rtol, equi_save...)
            ints[i,j], errs[i,j], equi_save = r.value
            times[i,j] = r.time
            npt1[i,j] = equi_save.npt1
            npt2[i,j] = equi_save.npt2
        end
        @info "npt1=$(equi_save.npt1), npt2=$(equi_save.npt2)"
        write_h5("DOS_scaling_equispace_$(id).h5", etas[1:j], omegas, ints[:,1:j], errs[:,1:j], times[:,1:j])
        write_h5_npt("DOS_scaling_equispace_npt_$(id).h5", etas[1:j], omegas, npt1[:,1:j], npt2[:,1:j])
    end
end

function adaptive_scaling()
    H, omegas, etas, t, id, atol, rtol, ints, errs, times = initialize_data()

    for (j, eta) in enumerate(etas)
        @info "starting log2(eta)=$(log2(eta))"
        Σ = EtaSelfEnergy(eta)
        for (i, omega) in ProgressBar(enumerate(omegas))
            D = DOSIntegrand(H, Σ, omega)
            r = @timed AutoBZ.iterated_integration(D, t; atol=atol, rtol=rtol)
            ints[i,j], errs[i,j] = r.value
            times[i,j] = r.time
        end
        write_h5("DOS_scaling_adaptive_$(id).h5", etas[1:j], omegas, ints[:,1:j], errs[:,1:j], times[:,1:j])
    end
end

function auto_adaptive_scaling(; eatol=0.0, ertol=1.0)
    H, omegas, etas, t, id, atol, rtol, ints, errs, times = initialize_data()
    equi_save = (npt1=0, pre1=Tuple{eltype(H),Int}[], npt2=0, pre2=Tuple{eltype(H),Int}[])
    for (j, eta) in enumerate(etas)
        Σ = EtaSelfEnergy(eta)
        for (i, omega) in enumerate(omegas)
            D = DOSIntegrand(H, Σ, omega)
            int_, = AutoBZ.automatic_equispace_integration(D, t; atol=eatol, rtol=ertol, equi_save...)
            r = @timed AutoBZ.iterated_integration(D, t; atol=max(atol, rtol*norm(int_)), rtol=0.0)
            ints[i,j], errs[i,j] = r.value
            times[i,j] = r.time
        end
        write_h5("DOS_scaling_auto_adaptive_$(id).h5", etas[1:j], omegas, ints[:,1:j], errs[:,1:j], times[:,1:j])
    end
end

function plot_scaling()
    H, omegas, etas, t, id, atol, rtol, ints, errs, times = initialize_data()
    plt = plot(; scale=:log10, xguide="η (eV)", yguide="Wall time (s)", title="DOS scaling for $(length(omegas)) frequencies, atol=$atol, rtol=$rtol", legend=:bottomleft)

    aetas, aomegas, ints, errs, atimes = read_h5("DOS_scaling_adaptive_$(id).h5")
    plot!(plt, aetas[2:end], vec(sum(atimes[:,2:end], dims=1)); color=1, markershape=:x, label="adaptive")
    plot!(plt, aetas[2:end], x -> 5e-1*log(1/x)^3; color=1, ls=:dash, label="O(log(1/η)³)")

    eetas, eomegas, ints, errs, etimes = read_h5("DOS_scaling_equispace_$(id).h5")
    plot!(plt, eetas[2:end], vec(sum(etimes[:,2:end], dims=1)); color=2, markershape=:x, label="equispace")
    plot!(plt, eetas[2:end], x -> 1e-4/x^3; color=2, ls=:dash, label="O(1/η³)")
    savefig("DOS_scaling.png")

    plt = plot(; yscale=:log10, xguide="ω (eV)", yguide="Wall time (s)", title="DOS adaptive scaling (ω resolved), atol=$atol, rtol=$rtol", legend=:bottomright)
    for (j, eta) in enumerate(aetas)
        j == 1 && continue # compilation messes up timing
        plot!(plt, aomegas, atimes[:,j]; color=1, label="log2(η)=$(log2(eta))", alpha=j/length(aetas))
    end
    savefig("DOS_scaling_adaptive.png")
    
    plt = plot(; yscale=:log10, xguide="ω (eV)", yguide="Wall time (s)", title="DOS equispace scaling (ω resolved), atol=$atol, rtol=$rtol", legend=:bottomright)
    for (j, eta) in enumerate(eetas)
        j == 1 && continue # compilation messes up timing
        plot!(plt, eomegas, etimes[:,j]; color=2, label="log2(η)=$(log2(eta))", alpha=j/length(eetas))
    end
    savefig("DOS_scaling_equispace.png")
    
    eetas, eomegas, npt1, npt2 = read_h5_npt("DOS_scaling_equispace_npt_$(id).h5")
    plt = plot(; yscale=:log10, xguide="ω (eV)", yguide="k points used", title="DOS equispace scaling (ω resolved), atol=$atol, rtol=$rtol", legend=:bottomright)
    for (j, eta) in enumerate(eetas)
        j == 1 && continue # compilation messes up timing
        plot!(plt, eomegas, npt1[:,j]; color=2, label="log2(η)=$(log2(eta))", alpha=j/length(eetas))
    end
    savefig("DOS_npt_equispace.png")
end

function plot_result()
    H, omegas, etas, t, id, atol, rtol, ints, errs, times = initialize_data()
    
    plt = plot(; xguide="ω (eV)", yguide="DOS(ω)", title="DOS adaptive result, atol=$atol, rtol=$rtol", legend=:topleft)
    aetas, aomegas, aints, errs, times = read_h5("DOS_scaling_adaptive_$(id).h5")
    for (j, eta) in enumerate(aetas)
        plot!(plt, aomegas, aints[:,j]; color=1, label="log2(η)=$(log2(eta))", alpha=j/length(aetas))
    end
    savefig("DOS_result_adaptive.png")

    plt = plot(; xguide="ω (eV)", yguide="DOS(ω)", title="DOS equispace result, atol=$atol, rtol=$rtol", legend=:topleft)
    eetas, eomegas, eints, errs, times = read_h5("DOS_scaling_equispace_$(id).h5")
    for (j, eta) in enumerate(eetas)
        plot!(plt, eomegas, eints[:,j]; color=2, label="log2(η)=$(log2(eta))", alpha=j/length(eetas))
    end
    savefig("DOS_result_equispace.png")
end

function plot_error()
    H, omegas, etas, t, id, atol, rtol, ints, errs, times = initialize_data()

    aetas, aomegas, aints, aerrs, times = read_h5("DOS_scaling_adaptive_$(id).h5")
    eetas, eomegas, eints, eerrs, times = read_h5("DOS_scaling_equispace_$(id).h5")
    
    # validation dataset
    tetas, tomegas, tints, errs, times = read_h5("old_DOS_scaling_adaptive_atol-5_rtol-Inf.h5")
    
    tomegas == aomegas || return
    plt = plot(; xguide="ω (eV)", yguide="|adaptive-true|", title="DOS absolute error, atol=$atol, rtol=$rtol", yscale=:log10, legend=:topright, ylims=(1e-5, 1e0))
    hst = plot(; xguide="log10(absolute error)", yguide="counts", title="DOS adaptive error distribution", legend=:topleft)
    hst2 = plot(; xguide="log10(reported error)", yguide="counts", title="DOS adaptive error distribution", legend=:topleft)
    for (j, (eta, eta_)) in enumerate(zip(tetas, aetas))
        @assert eta == eta_
        err_ = abs.(tints[:,j] .- aints[:,j])
        scatter!(plt, tomegas, err_; color=1, label="log2(η)=$(log2(eta))", alpha=j/length(etas), markershape=:x)
        histogram!(hst, log10.(err_); label="log2(η)=$(log2(eta))", color=j, alpha=j/length(etas))
        histogram!(hst2, log10.(abs.(aerrs[:,j])); label="log2(η)=$(log2(eta))", color=j, alpha=j/length(etas))
    end
    plot!(plt, tomegas, _ -> atol; color=:red, label="atol")
    savefig(plt,"DOS_error_adaptive.png")
    vline!(hst, [log10(atol)]; color=:red, label="atol")
    savefig(hst, "DOS_error_adaptive_distribution.png")
    vline!(hst2, [log10(atol)]; color=:red, label="atol")
    savefig(hst2, "DOS_error_adaptive_distribution_reported.png")

    tomegas == eomegas || return
    plt = plot(; xguide="ω (eV)", yguide="|equispace-true|", title="DOS absolute error, atol=$atol, rtol=$rtol", yscale=:log10, legend=:topright, ylims=(1e-5, 1e0))
    hst = plot(; xguide="log10(absolute error)", yguide="counts", title="DOS equispace error distribution", legend=:topleft)
    hst2 = plot(; xguide="log10(reported error)", yguide="counts", title="DOS adaptive error distribution", legend=:topleft)
    for (j, (eta, eta_)) in enumerate(zip(tetas, eetas))
        @assert eta == eta_
        err_ = abs.(tints[:,j] .- eints[:,j])
        scatter!(plt, tomegas, err_; color=2, label="log2(η)=$(log2(eta))", alpha=j/length(etas), markershape=:x)
        histogram!(hst, log10.(err_); label="log2(η)=$(log2(eta))", color=j, alpha=j/length(etas))
        histogram!(hst2, log10.(abs.(eerrs[:,j])); label="log2(η)=$(log2(eta))", color=j, alpha=j/length(etas))
    end
    plot!(plt, tomegas, _ -> atol; color=:red, label="atol")
    savefig(plt,"DOS_error_equispace.png")
    vline!(hst, [log10(atol)]; color=:red, label="atol")
    savefig(hst, "DOS_error_equispace_distribution.png")
    vline!(hst2, [log10(atol)]; color=:red, label="atol")
    savefig(hst2, "DOS_error_equispace_distribution_reported.png")
end

initialize_order() = 3:7

function oadaptive_scaling()
    H, omegas, etas, t, id, atol, rtol, ints, errs, times = initialize_data()
    orders = initialize_order()
    for o in orders
        @info "using order $o GK rule"
        for (j, eta) in enumerate(etas)
            @info "starting log2(eta)=$(log2(eta))"
            Σ = EtaSelfEnergy(eta)
            for (i, omega) in ProgressBar(enumerate(omegas))
                D = DOSIntegrand(H, Σ, omega, μ)
                r = @timed iterated_integration(D, t; order=o, atol=atol, rtol=rtol)
                ints[i,j], errs[i,j] = r.value
                times[i,j] = r.time
            end
            write_h5("DOS_scaling_adaptive_o$(o)_$(id).h5", etas[1:j], omegas, ints[:,1:j], errs[:,1:j], times[:,1:j])
        end
    end
end

function plot_order()
    H, omegas, etas, t, id, atol, rtol, ints, errs, times = initialize_data()
    orders = initialize_order()
    plt = plot(; xscale=:log10, xguide="η (eV)", yguide="Wall time (s)", title="DOS scaling for $(length(omegas)) frequencies, atol=$atol, rtol=$rtol", legend=:bottomleft)

    for (i,o) in enumerate(orders)
        aetas, aomegas, ints, errs, atimes = read_h5("DOS_scaling_adaptive_o$(o)_$(id).h5")
        plot!(plt, aetas[2:end], vec(sum(atimes[:,2:end], dims=1)); color=i, markershape=:x, label="GK $o")
        # plot!(plt, aetas[2:end], x -> 5e-1*log(1/x)^3; color=1, ls=:dash, label="O(log(1/η)³)")

        plt_ = plot(; yscale=:log10, xguide="ω (eV)", yguide="Wall time (s)", title="DOS GK $o scaling (ω resolved), atol=$atol, rtol=$rtol", legend=:bottomright)
        for (j, eta) in enumerate(aetas)
            j == 1 && continue # compilation messes up timing
            plot!(plt_, aomegas, atimes[:,j]; color=i, label="log2(η)=$(log2(eta))", alpha=j/length(aetas))
        end
        savefig(plt_, "DOS_order$(o).png")
    end
    savefig(plt, "DOS_order.png")

end

function plot_order_error()
    H, omegas, etas, t, id, atol, rtol, ints, errs, times = initialize_data()
    orders = initialize_order()

    # validation dataset
    tetas, tomegas, tints, errs, times = read_h5("DOS_scaling_adaptive_atol-5_rtol-Inf.h5")
    @show tetas 
    plt = plot(; xguide="Evaluation points grouped by eta (smaller -> darker)", yguide="|adaptive-true|", title="DOS absolute error, atol=$atol, rtol=$rtol", yscale=:log10, legend=:bottomleft, ylims=(1e-5, 1e0), xticks=[])
    for (i,o) in enumerate(orders)
        aetas, aomegas, aints, errs, times = read_h5("DOS_scaling_adaptive_o$(o)_$(id).h5")
        @show aetas 
        tomegas == aomegas || return
        for (j, (eta, eta_)) in enumerate(zip(tetas, aetas))
            @assert eta == eta_
            label = ifelse(j == length(tetas), "GK $o", "")
            scatter!(plt, (1:length(tomegas)) .+ (j-1 + length(tetas)*(i-1))*length(tomegas), abs.(tints[:,j] .- aints[:,j]); color=i, label=label, alpha=(j/length(tetas))^3, markershape=:x)
        end
    end
    plot!(plt, [1, (length(tetas)*length(orders)+1)*length(tomegas)], _ -> atol; color=:red, label="atol")
    savefig("DOS_error_order.png")

end
using Plots

function make_plot()
    nbfull = Demos.read_h5_to_nt("OC_results_noberry_full_equispace_10kpts_2scatter_1K.h5")
    nbinter = Demos.read_h5_to_nt("OC_results_noberry_inter_equispace_10kpts_2scatter_1K.h5")
    nbintra = Demos.read_h5_to_nt("OC_results_noberry_intra_equispace_10kpts_2scatter_1K.h5")
    bfull = Demos.read_h5_to_nt("OC_results_berry_full_equispace_10kpts_2scatter_1K.h5")
    binter = Demos.read_h5_to_nt("OC_results_berry_inter_equispace_10kpts_2scatter_1K.h5")
    bintra = Demos.read_h5_to_nt("OC_results_berry_intra_equispace_10kpts_2scatter_1K.h5")
    plt = plot(; xguide="Ω (eV)", yguide="σ₁₁ (Ω⁻¹cm⁻¹)", yscale=:log10, title="SrVO3 OC @10 kpts/dim")
    plot!(nbfull.Omega, 616.57354879193*map(real∘first, nbfull.OC); color=1, label="vw/noA/full")
    plot!(nbinter.Omega, 616.57354879193*map(real∘first, nbinter.OC); color=2, label="vw/noA/inter")
    plot!(nbintra.Omega, 616.57354879193*map(real∘first, nbintra.OC); color=3, label="vw/noA/intra")
    plot!(bfull.Omega, 616.57354879193*map(real∘first, bfull.OC); color=4, label="vw/withA/full")
    plot!(binter.Omega, 616.57354879193*map(real∘first, binter.OC); color=5, label="vw/withA/inter")
    plot!(bintra.Omega, 616.57354879193*map(real∘first, bintra.OC); color=6, label="vw/withA/intra")
    return plt
end
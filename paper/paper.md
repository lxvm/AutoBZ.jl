---
title: 'AutoBZ.jl: automatic, adaptive Brillouin zone integration of response functions using Wannier interpolation'
tags:
  - Julia
  - electronic structure theory
  - solid-state
  - computational materials science
  - Brillouin-zone integration
  - Optical conductivity
authors:
  - name:
      given-names: Lorenzo
      non-dropping-particle: Van
      surname: Munoz
    orcid: 0000-0003-0807-5034
    corresponding: true
    affiliation: 1
  - name: Sophie Beck
    orcid: 0000-0002-9336-6065
    affiliation: 2
  - name: Jason Kaye
    orcid: 0000-0001-8045-6179
    affiliation: "2, 3"
affiliations:
 - name: Department of Physics, Massachusetts Institute of Technology, 77 Massachusetts Avenue, Cambridge, MA 02139, USA
   index: 1
 - name: Center for Computational Quantum Physics, Flatiron Institute, 162 5th Avenue, New York, NY 10010, USA
   index: 2
 - name: Center for Computational Mathematics, Flatiron Institute, 162 5th Avenue, New York, NY 10010, USA
   index: 3
date: XXX 2024
bibliography: paper.bib
---

# Summary


AutoBZ.jl is a modular Julia package implementing efficient algorithms for Brillouin zone (BZ) integration, a fundamental step in the calculation of physical observables such as the density of states and the optical conductivity.
Our BZ integration methods, described in Refs. [@kayeAutomaticHighorderAdaptive2023] and [@VanMunoz_et_al:2024], are high-order accurate, automatically convergent to a user-specified error tolerance, and if needed, adaptive in momentum space.
This allows the resolution of low temperature properties of strongly interacting systems, using many-body methods such as dynamical mean-field theory[@georgesDynamicalMeanfieldTheory1996a] in which frequency-dependent electronic self-energies may produce scattering rates in the sub-meV regime. The corresponding low temperature regions of phase diagrams are typically out of reach using traditional integration algorithms, which struggle to resolve localized features in momentum space.
AutoBZ.jl can also be used to compute ground state (i.e. $T=0$ K) properties of tight-binding models, typically derived from localized Wannier functions, with a given artificial broadening.
Designed using open-source software principles [@bezansonJuliaFreshApproach2017], AutoBZ.jl serves as an extensible framework for computational research on material properties, and a flexible toolbox with which to compute a broad range of quantities using Wannier interpolation and BZ integration [@mostofiWannier90ToolObtaining2008].
We expect it to become a widely used tool in the electronic structure community, providing accurate comparisons with experimental spectra, and a robust, automated approach for high-throughput screenings and machine learning of material properties.

# Statement of need

Most open-source software packages used for density functional theory plus dynamical mean-field theory, including those compatible with Wannier90 (see, e.g., Refs. [@aichhornTRIQSDFTToolsTRIQS2016; @Romero_et_al:2020; @Shinaoka_et_al:2021; @Singh_et_al:2021]), employ simple uniform grids for BZ integration, despite the fact that BZ integrands may be highly localized, e.g., near energy isosurfaces for the Green's function.
However, these details of electronic structure may sensitively control downstream observables, so it is crucial that BZ integrals be computed in a resolved manner in material-realistic calculations. 
In practice, this requires using dense uniform integration grids, which become compute or memory-limited in low temperature calculations involving scattering rates approaching the meV scale.
Furthermore, validating uniform integration methods requires careful convergence testing which is not always prioritized, sometimes leading to under-resolved results with spurious features.
The algorithms in AutoBZ.jl, which include an optimized uniform grid integration scheme for larger scattering rates, automate convergence testing to provide results to a user-specified error tolerance.
For low temperature calculations, our adaptive integration algorithm has only polylogarithmic computational complexity with respect to the scattering rate, superior to the polynomial rates of alternative tree-based adaptive methods.
These advancements will be crucial for the development of next-generation quantum impurity solvers and the accurate characterization of spectral features for low-temperature phenomena in condensed matter physics.

# Design principles

AutoBZ.jl is developed in a modular, Julian fashion involving several components required for integration [@vanmunozAutoBZCoreJlWannier2023] and interpolation [@vanmunozHChebInterpJlMultidimensional2023; @vanmunozFourierSeriesEvaluatorsJlFourier2023] which may be of independent interest in other scientific computing applications.
It also contains an extension to SymmetryReduceBZ.jl [@jorgensenGeneralAlgorithmCalculating2022a] for optimizations involving the lattice symmetry group, including an implementation of a symmetric Monkhorst-Pack grid using the algorithm of Ref. [@hartRobustAlgorithmKpoint2019].
We use the CommonSolve.jl interface to promote interoperability with existing packages.
For example, we provide a routine to compute the electron density which can easily be combined with NonlinearSolve.jl[@pal2024nonlinearsolve] to determine the chemical potential.
AutoBZ.jl can be called from MATLAB and Python, and includes file-based interfaces to read Wannier90 output, such as Hamiltonian and position operator matrix elements, as well as frequency-dependent self-energy data determined either phenomenologically or using a quantum many-body framework.
The modular design of AutoBZ.jl simplifies the addition of new algorithms and problem types, and its interoperatibility and well-documented API enables its use as a scripting tool for many research problems.


![Calculations of various physical observables for a 3-band model of $t_{2g}$ orbitals in
the cubic perovskite SrVO3 at a temperature of 24 K, using a Fermi liquid scaling for the scattering rate, which is 1 meV. Panel (a) shows the density of states (DOS) as a function of frequency and panel (b) shows the optical conductivity as a function of excitation frequency at a chemical potential of 12.5 eV.
AutoBZ.jl
was used to compute the observables at automatically-selected interpolation nodes determined by HChebinterp.jl, yielding a result which can be evaluated on the full domain. We compare the resolved results obtained using adaptive integration with the result of uniform grid integration with $N_k$ momentum space points per dimension. \label{fig:observables}](figure.png)

# Acknowledgements

We thank Steven G. Johnson, Fabian Kugler, and Alex Barnett for helpful feedback and discussions.
The Flatiron Institute is a division of the Simons Foundation. 

# References

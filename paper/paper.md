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


AutoBZ.jl is a modular Julia package implementing efficient algorithms for Brillouin zone (BZ) integration, a fundamental step in the calculation of response functions such as the density of states and optical conductivity.
Our BZ integration methods, described in Refs. [@kayeAutomaticHighorderAdaptive2023] and [@VanMunoz_et_al:2024], are high-order accurate, automatically convergent to a user-specified error tolerance, and if needed, adaptive in momentum space.
This allows sub-meV energy resolution that is a common requirement in many-body methods for strongly interacting systems, such as dynamical mean-field theory[@georgesDynamicalMeanfieldTheory1996a] where frequency-dependent electronic self-energies may attain scattering rates as low as 0.1 meV. This regime is typically out of reach using traditional integration algorithms, which struggle to resolve localized features in momentum space.
AutoBZ.jl can also be used to compute ground state (i.e. T=0K) properties of tight-binding models, typically derived from localized Wannier functions, with a given artificial broadening.
Designed using open-source software principles [@bezansonJuliaFreshApproach2017], AutoBZ.jl serves as an extensible framework for research into computational materials response phenomena, and a flexible toolbox with which to compute a broad range of quantities using Wannier interpolation and BZ integration [@mostofiWannier90ToolObtaining2008].
We expect it to have a broad impact on the electronic structure community, providing accurate comparisons with experimental spectra, and a robust, automated approach for high-throughput screenings and machine learning of materials properties.

<!---
and our goal is to use it to study strongly
interacting systems
with sufficient energy resolution, i.e. sub-meV, to elucidate the various
effects of interactions, dispersion, and spin-orbit coupling. In particular, we
believe the DMFT  community will benefit
from this package, either as a post-processing tool for experimental
predictions, such as the calculation presented in \autoref{fig:oc}, or as an
inner-loop calculation, such as for ensuring charge self-consistency.
-->

# Statement of need

<!---
In recent years, DFT codes combined with tools such as Wannier90
[@mostofiWannier90ToolObtaining2008]
have enabled high-throughput materials searches by robustly calculating the
electronic structure of crystalline solids from first principles
[@vitaleAutomatedHighthroughputWannierisation2020]. To
compare theory and experiment, the last step in predicting the electronic and
optical properties of these solids is calculating Brillouin-zone (BZ) integrals
to obtain quantities
such as the dielectric function, the density of states, and the Hall
conductivity.
-->

Most electronic structure packages, including those compatible with Wannier90 [@tsirkinHighPerformanceWannier2021; @aichhornTRIQSDFTToolsTRIQS2016], employ simple uniform integration grids, despite the typical localization of BZ integrands, e.g., near Fermi surfaces in the case of the Green's function.
However, these details of electronic structure may sensitively control downstream observables, so it is crucial that BZ integrals be computed in a resolved manner in material-realistic calculations. 
In practice, this requires using dense uniform integration grids, which become compute or memory-limited in low temperature calculations involving scattering rates approaching the meV scale.
Furthermore, validating uniform integration methods requires careful convergence testing which is often neglected, sometimes leading to under-resolved results with spurious features.
The algorithms in AutoBZ.jl, which include a uniform grid integration scheme for larger scattering rates, automate convergence testing to provide results to a user-specified error tolerance.
For low temperature calculations, our adaptive integration algorithm has only polylogarithmic computational complexity with respect to the scattering rate, superior to the polynomial rates of alternative tree-based adaptive methods.
These advancements will be crucial for the development of next-generation quantum impurity solvers and the accurate characterization of spectral features for scientifically important low-temperature phenomena in condensed matter physics.

# Design principles

AutoBZ.jl is developed in a modular, Julian fashion involving several components required for integration [@vanmunozAutoBZCoreJlWannier2023] and interpolation [@vanmunozHChebInterpJlMultidimensional2023; @vanmunozFourierSeriesEvaluatorsJlFourier2023] which may be of independent interest in other scientific computing applications.
It also contains an extension to SymmetryReduceBZ.jl [@jorgensenGeneralAlgorithmCalculating2022a] for optimizations involving the lattice symmetry group, including an implementation of a symmetric Monkhorst-Pack grid using the algorithm of Ref. [@hartRobustAlgorithmKpoint2019].
We use the CommonSolve.jl interface to promote interoperability with existing packages.
For example, we provide a routine to compute the electron density which can easily be combined with NonlinearSolve.jl[@pal2024nonlinearsolve] to determine the chemical potential.
AutoBZ.jl can be called from MATLAB and Python, and includes file-based interfaces to read Wannier90 output, such as Hamiltonian and position operator matrix elements, as well as frequency-dependent self-energy data determined either phenomenologically or using a quantum many-body framework.
The modular design of AutoBZ.jl simplifies the addition of new algorithms and problem types, and its interoperatibility and well-documented API enables its use as a scripting tool for many research problems.


![An optical conductivity calculation for a 3-band model of $t_{2g}$ orbitals in
the cubic perovskite SrVO3 across a geometric
series of temperatures such that the scattering rate is halved each time the
temperature is decreased, reaching a minimum value of 0.2 meV. 
The inset shows the $1/T^2$ scaling of the DC conductivity expected for Fermi liquids.
AutoBZ.jl
was used to compute the conductivity, which was interpolated by HChebinterp.jl
with parallelization of both the integration and interpolation. \label{fig:oc}](oc_fermiliquid.png)

# Acknowledgements

We thank Steven G. Johnson, Fabian Kugler, and Alex Barnett for helpful feedback and discussions.
The Flatiron Institute is a division of the Simons Foundation. 

# References

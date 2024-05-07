---
title: 'AutoBZ.jl: automatic, adaptive Brillouin-zone integration of response functions using Wannier interpolation'
tags:
  - Julia
  - electronic structure theory
  - solid-state
  - computational materials science
  - Brillouin-zone integration
  - Optical conductivity
authors:
  - name: Lorenzo Van Munoz
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


AutoBZ.jl is a modular software package developed to ~~We developed AutoBZ.jl to~~ explore efficient algorithms ~~and codes for the 
challenging, nearly singular~~ for the computation of Brillouin zone (BZ) integrals, which are a fundamental and ubiquitous step in electronic structure calculations in the computation of physical observables. ~~commonly occur
in response function calculations in solid-state physics.~~
Designed on open-source software principles and written in Julia
[@bezansonJuliaFreshApproach2017], our package provides high-order accurate and
automatically-converging algorithms to compute observables such as, e.g. the density of states or optical conductivity ~~at challenging sub-meV energy scales~~ using adaptive integration algorithms
proposed in Ref. [@kayeAutomaticHighorderAdaptive2023].
Moreover, AutoBZ.jl serves as an extensible framework for future projects on computational
materials response phenomena, providing a flexible toolbox to calculate a broad range of quantities involving BZ integrals.
The package operates in the setting of a small Hamiltonian in a Wannier basis, i.e. with a rapid decay enabling efficient Fourier interpolation in reciprocal space.
For such cases it enables users to compute BZ integrals based on the electronic ground state (i.e. T=0K) represented as tight-binding models and an artificial broadening automatically and to a user-specified error tolerance.
Additionally, it gives access to challenging sub-meV energy scales when combined with frequency-dependent electronic self-energies based on many-body methods for strongly interacting systems such as dynamical mean-field theory[@georgesDynamicalMeanfieldTheory1996a].
We expect AutoBZ.jl to have a broad impact on the electronic structure
community by providing, for example, accurate benchmarks for comparison with
experimental spectra, and a robust, automated approach for high-throughput
screenings and machine learning of materials properties.

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

In recent years, DFT codes combined with tools such as Wannier90
[@mostofiWannier90ToolObtaining2008]
have enabled high-throughput materials searches by robustly calculating the
electronic structure of crystalline solids from first principles
[@vitaleAutomatedHighthroughputWannierisation2020]. To
compare theory and experiment, the last step in predicting the electronic and
optical properties of these solids is calculating Brillouin-zone (BZ) integrals
to obtain quantities
such as the dielectric function, the density of states, and the Hall
conductivity. Often the details of the electronic structure may very sensitively
control the resonant features of these observable quantities, which makes it
crucial that this final step in many material-realistic calculations be as
accurate as possible and reflect underlying theoretical predictions
[@kratzerBasicsElectronicStructure2019].
Most existing BZ integration libraries that compute
the optical conductivity, including Refs.
[@tsirkinHighPerformanceWannier2021; @aichhornTRIQSDFTToolsTRIQS2016],
are restricted to using uniform integration grids despite the fact the
conductivity integrand may be nearly singular at resonances.
In practice, this means integration grids must become very dense to attain good
accuracy and quickly become time or memory-limited even for modest problems.
Our work employs automatic and adaptive integration algorithms that provide the
confidence to perform accurate response function calculations with enough speed
to access energy scales that were previously intractable.

# Design principles

Our package was developed in a modular, Julian fashion with various components
for integration [@vanmunozAutoBZCoreJlWannier2023] and interpolation
[@vanmunozHChebInterpJlMultidimensional2023; @vanmunozFourierSeriesEvaluatorsJlFourier2023]
that may be independently useful.
We also include a Julia package extension to
SymmetryReduceBZ.jl [@jorgensenGeneralAlgorithmCalculating2022a]
to optimize our integration using the symmetry group of a lattice, including an
implementation of a symmetric Monkhorst-Pack grid using the algorithm in Ref.
[@hartRobustAlgorithmKpoint2019], as well as a routine to calculate
the electron density that can easily be combined with NonlinearSolve.jl
[@pal2024nonlinearsolve] as a chemical potential finder.
AutoBZ.jl can be called from MATLAB and Python and it has file-based
interfaces to read output from the Wannier90 code such as the Hamiltonian and the position operator matrix elements, as well as frequency-dependent self-energy
data, that is either phenomenological or calculated within a many-body framework. The benefits of this modular design are that contributing new algorithms
and problem types to the code base is simplified with well-documented APIs and
that our package's intentional interoperatibility enables its use as a
scripting tool for many interesting research problems.

![An optical conductivity calculation for a 3-band model of $t_{2g}$ orbitals in
the cubic perovskite SrVO3 across a geometric
series of temperatures such that the scattering rate is halved each time the
temperature is decreased, reaching a minimum value of 0.2 meV. 
The inset shows the $1/T^2$ scaling of the DC conductivity expected for Fermi liquids.
AutoBZ.jl
was used to compute the conductivity, which was interpolated by HChebinterp.jl
with parallelization of both the integration and interpolation. \label{fig:oc}](oc_fermiliquid.png)

# Acknowledgements

We thank Steven G. Johnson and Fabian Kugler for helpful discussions.
The Flatiron Institute is a division of the Simons Foundation. 

# References

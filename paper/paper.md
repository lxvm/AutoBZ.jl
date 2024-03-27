---
title: 'AutoBZ.jl: automatic, adaptive Brillouin-zone integration of Wannier-interpolated response functions'
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

In recent years, open source DFT codes combined with tools such as Wannier90
have enabled high-throughput materials searches by robustly calculating the
electronic structure of many metals and crystals from first principles. To
compare theory and experiment, the last step in predicting the electronic and
optical properties of these solids is calculating integrals to obtain quantities
such as the dielectric function, the density of states (DOS), and the Hall
conductivity. Often the details of the electronic structure may very sensitively
control the resonant features of these observable quantities, which makes it
crucial that this final step in many material-realistic calculations be as
accurate as possible and reflect underlying theoretical predictions. We
developed AutoBZ.jl to explore efficient algorithms and codes for the
challenging, nearly singular integrals that occur in response function
calculations that commonly arise in problems solid-state physics. Designed on
open-source software principles and written in Julia, our package enables
high-order accurate and parallelizable optical conductivity and DOS calculations
at challenging sub-meV energy scales and serves as an extensible framework for
future projects on materials response phenomena.

# Statement of need

![The optical conductivity of SrVO3.\label{fig:oc}](oc_fermiliquid.png)
and referenced from text using \autoref{fig:oc}.

[@tsirkinHighPerformanceWannier2021]

# Acknowledgements

We thank ... for helpful discussions.
The Flatiron Institute is a division of the Simons Foundation. 

# References
"""
This Julia package provides routines for multi-dimensional Brillouin zone (BZ)
integration for applications of Wannier interpolation to the calculation of density of
states, chemical potential, and optical conductivity with self energies.
It uses algorithms which automatically compute BZ integrals to a
specified error tolerance by resolving smooth yet highly localized integrands.

See [AutoBZCore.jl](https://lxvm.github.io/AutoBZCore.jl/dev/) on how to create
custom integrands and for more details on the algorithms.

In many-body Green's function methods, BZ integrands are localized at a scale
determined by a non-zero, but possibly small, system- and temperature-dependent
scattering rate. For example, the single-particle retarded Green's function of
an electronic system for frequency ``\\omega`` and reciprocal space vector
``\\bm{k}`` with chemical potential ``\\mu``, Hermitian Hamiltonian matrix
``H(\\bm{k})``, and self-energy matrix ``\\Sigma(\\omega)``, which is given by
```math
G(\\omega) = \\int_{\\text{BZ}} d\\bm{k}\\ \\operatorname{Tr} \\left[ (\\hbar\\omega - H(\\bm{k}) - \\Sigma(\\omega))^{-1} \\right]
```
is localized about the manifold defined by ``\\det(\\hbar\\omega - H(\\bm{k}))=0`` (i.e.
the Fermi surface when ``\\hbar\\omega=\\mu``) by a scattering rate depending on
``\\operatorname{Im}\\ \\Sigma(\\omega)``.

## Package features

* Iterated adaptive integration (IAI) with nested calls to
  [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl)
    * Algorithm with logarithmic complexity for increasingly localized integrands
    * Irreducible Brillouin zone (IBZ) integration for the cubic lattice
* Equispace integration (PTR) as described by Kaye et al. [^1]
    * Automatic algorithm that refines ``k``-grid to meet requested error
* Support for Wannier-interpolated integrands
    * User-defined integrands based on Bloch Hamiltonians
    * Density of states (DOS) calculations
    * Transport calculations based on
        [TRIQS DFTTools](https://triqs.github.io/dft_tools/latest/guide/transport.html)
        * Calculation of transport function and kinetic coefficients
        * Option to separate intra-band and inter-band contributions
        * Parallelized calculations available through `batchsolve` interface of AutoBZCore.jl
    * [Wannier90](http://www.wannier.org/)-based parsers Hamiltonians
      (`*_hr.dat` files) and position operators (`*_r.dat` files)
    * Automated interpolation for frequency-dependent self-energy data in text files, using
      [EquiBaryInterp.jl](https://github.com/lxvm/EquiBaryInterp.jl) and
      [HChebInterp.jl](https://github.com/lxvm/HChebInterp.jl).
* IBZ integration for arbitrary symmetry groups (via an interface to
  [SymmetryReduceBZ.jl](https://github.com/jerjorg/SymmetryReduceBZ.jl))

[^1]: [Kaye et al. "Automatic, high-order, and adaptive algorithms for Brillouin zone integration"](http://arxiv.org/abs/2211.12959)
"""
module AutoBZ

using LinearAlgebra
using LinearAlgebra: checksquare
using Logging

using StaticArrays
using Reexport
@reexport using FourierSeriesEvaluators
@reexport using AutoBZCore

import FourierSeriesEvaluators: period, frequency, allocate, contract!, evaluate!, nextderivative, show_dims, show_details
import AutoBZCore: symmetrize_, IntegralAlgorithm, AutoBZAlgorithm, AbstractBZ, interior_point
import CommonSolve: init, solve!

using FourierSeriesEvaluators: FourierWorkspace, freq2rad
using EquiBaryInterp: LocalEquiBaryInterp
using BaryRational: aaa
using HChebInterp: hchebinterp
using FastLapackInterface: LUWs, EigenWs, HermitianEigenWs

export AbstractSelfEnergy
export AbstractWannierInterp
export AbstractGauge, Wannier, Hamiltonian
export AbstractGaugeInterp, gauge
export AbstractCoordinate, Lattice, Cartesian
export AbstractCoordInterp, coord
export AbstractVelocityComponent, Whole, Inter, Intra
export AbstractVelocityInterp, vcomp, shift!
include("definitions.jl")

export diag_inv, tr_inv, tr_mul, herm, commutator
include("linalg.jl")
export LinearSystemProblem, LUFactorization, JLInv
include("linearsystem.jl")
export EigenProblem, LAPACKEigen, LAPACKEigenH, JLEigen
include("eigen.jl")
export TraceInverseProblem, JLTrInv, LinearSystemTrInv, EigenTrInv
include("trinv.jl")
export fermi, fermi′, fermi_window, fermi_window_limits
include("fermi.jl")

export HamiltonianInterp
export BerryConnectionInterp
export GradientVelocityInterp
export CovariantVelocityInterp
export MassVelocityInterp
include("interp.jl")

export EtaSelfEnergy, ConstScalarSelfEnergy, ScalarSelfEnergy, DiagonalSelfEnergy, MatrixSelfEnergy
include("self_energies.jl")

export GlocSolver, DiagGlocSolver, TrGlocSolver, DOSSolver
export TransportFunctionSolver, TransportDistributionSolver
export KineticCoefficientSolver, OpticalConductivitySolver
export ElectronDensitySolver
export AuxTransportDistributionSolver, AuxKineticCoefficientSolver, AuxOpticalConductivitySolver
include("apps.jl")

export load_self_energy
include("self_energies_io.jl")

export SOCMatrix, SOCHamiltonianInterp
include("soc.jl")
include("SSymmetricCompact.jl")

export load_interp, load_autobz, load_wannier90_data
include("wannier90io.jl")


end

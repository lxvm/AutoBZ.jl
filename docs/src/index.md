# AutoBZ.jl documentation

This Julia package provides routines for multi-dimensional Brillouin zone (BZ)
integration of both generic and Wannier-interpolated integrands.
It aims to implement algorithms which automatically compute BZ integrals to a
specified error tolerance by resolving smooth yet highly localized integrands.

In many-body Green's function methods, BZ integrands are localized at a scale
determined by a non-zero, but possibly small, system- and temperature-dependent
scattering rate. For example, the single-particle retarded Green's function of
an electronic system for frequency ``\omega`` and reciprocal space vector ``k``
with chemical potential ``\mu``, Hermitian Hamiltonian matrix ``H(k)``, and
self-energy matrix ``\Sigma(\omega)``, which is given by
```math
G(\omega) = \int_{\text{BZ}} dk\ \operatorname{Tr} \left[ (\hbar\omega + \mu - H(k) - \Sigma(\omega))^{-1} \right]
```
is localized about the manifold defined by ``\det(\omega + \mu - H(k))=0`` (i.e.
the Fermi surface when ``\omega=0``) by a scattering rate depending on
``\operatorname{Im}\ \Sigma(\omega)``.

To start using the package, see the [Workflow](@ref) and [Demos](@ref) sections.

## Package features

### Implemented
* Iterated adaptive integration (IAI) with nested calls to
  [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl)
    * Algorithm with logarithmic complexity for increasingly localized integrands
    * Irreducible Brillouin zone (IBZ) integration for the cubic lattice
* Equispace integration (PTR) as described by Kaye et al. [^1]
    * Automatic algorithm that refines ``k``-grid to meet requested error
    * IBZ integration for arbitrary symmetry groups
* Support for Wannier-interpolated integrands in the Applications module
    * User-defined integrands based on Bloch Hamiltonians
    * Density of states (DOS) calculations
    * Transport calculations based on 
        [TRIQS DFTTools](https://triqs.github.io/dft_tools/latest/guide/transport.html)
        * Calculation of transport function and kinetic coefficients
        * Option to separate intra-band and inter-band contributions
        * Parallelized calculations available through [Jobs](@ref) module
    * [Wannier90](http://www.wannier.org/)-based parsers Hamiltonians
      (`*_hr.dat` files) and position operators (`*_r.dat` files)
    * Support for frequency-dependent self energy evaluators from data on
      uniform grids with the [EquiBaryInterp](@ref) module
* 1D adaptive Chebyshev interpolation in the [AdaptChebInterp](@ref) module
    using [FastChebInterp.jl](https://github.com/stevengj/FastChebInterp.jl)

### In progress
* IAI with IBZ integration for arbitrary symmetry groups (via an interface to
  [SymmetryReduceBZ.jl](https://github.com/jerjorg/SymmetryReduceBZ.jl))
* Interface to read frequency-dependent matrix-valued self-energy data

### More ideas
* Multi-dimensional adaptive Chebyshev interpolation, like
  [baobzi](https://github.com/flatironinstitute/baobzi)
* Globally adaptive IAI

## Notes

If you are an interested Python user, see the [Python](@ref) section

To see a poster showcasing calculations with the library, click this
[link](https://web.mit.edu/lxvm/www/slides/Lorenzo_VanMunoz_CCQ_intern_poster_2022.pdf)

For tree-adaptive integration (TAI) on the full BZ, see
[HCubature.jl](https://github.com/JuliaMath/HCubature.jl)

## Contact the developer

[Lorenzo Van Muñoz](https://web.mit.edu/lxvm/www/)

[^1]: [Kaye et al. "Automatic, high-order, and adaptive algorithms for Brillouin zone integration"](http://arxiv.org/abs/2211.12959)
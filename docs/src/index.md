# AutoBZ.jl documentation

This Julia package provides routines for multi-dimensional Brillouin zone (BZ)
integration of both generic and Wannier-interpolated integrands.

## Package features

To start using the package, see the [Workflow](@ref) and [Demos](@ref) sections

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

[^1]: [Kaye et al. "Automatic, high-order, and adaptive algorithms for Brillouin zone integration"](http://arxiv.org/abs/2211.12959)
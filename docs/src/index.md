# AutoBZ.jl documentation

This Julia package provides routines for multi-dimensional Brillouin zone (BZ)
integration of both generic and Wannier-interpolated integrands.
It aims to implement algorithms which automatically compute BZ integrals to a
specified error tolerance by resolving smooth yet highly localized integrands.

In many-body Green's function methods, BZ integrands are localized at a scale
determined by a non-zero, but possibly small, system- and temperature-dependent
scattering rate. For example, the single-particle retarded Green's function of
an electronic system for frequency ``\omega`` and reciprocal space vector
``\bm{k}`` with chemical potential ``\mu``, Hermitian Hamiltonian matrix
``H(\bm{k})``, and self-energy matrix ``\Sigma(\omega)``, which is given by
```math
G(\omega) = \int_{\text{BZ}} d\bm{k}\ \operatorname{Tr} \left[ (\hbar\omega - H(\bm{k}) - \Sigma(\omega))^{-1} \right]
```
is localized about the manifold defined by ``\det(\hbar\omega - H(\bm{k}))=0`` (i.e.
the Fermi surface when ``\hbar\omega=\mu``) by a scattering rate depending on
``\operatorname{Im}\ \Sigma(\omega)``.

To start using the package, see the [Workflow](@ref) and [Demos](@ref) sections.
It may also be helpful to see the [AutoBZCore.jl
documentation](https://lxvm.github.io/AutoBZCore.jl/dev/) since this package is
built on those interfaces.

## Package features

### Implemented
* Iterated adaptive integration (IAI) with nested calls to
  [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl)
    * Algorithm with logarithmic complexity for increasingly localized integrands
    * Irreducible Brillouin zone (IBZ) integration for the cubic lattice
* Equispace integration (PTR) as described by Kaye et al. [^1]
    * Automatic algorithm that refines ``k``-grid to meet requested error
* Support for Wannier-interpolated integrands in the Applications module
    * User-defined integrands based on Bloch Hamiltonians
    * Density of states (DOS) calculations
    * Transport calculations based on 
        [TRIQS DFTTools](https://triqs.github.io/dft_tools/latest/guide/transport.html)
        * Calculation of transport function and kinetic coefficients
        * Option to separate intra-band and inter-band contributions
        * Parallelized calculations available through `batchsolve` interface of AutoBZCore.jl
    * [Wannier90](http://www.wannier.org/)-based parsers Hamiltonians
      (`*_hr.dat` files) and position operators (`*_r.dat` files)
    * Automated interpolation for frequency-dependent data of [Self
      energies](@ref) in text files, using
      [EquiBaryInterp.jl](https://github.com/lxvm/EquiBaryInterp.jl) and
      [HChebInterp.jl](https://github.com/lxvm/HChebInterp.jl).
* IBZ integration for arbitrary symmetry groups (via an interface to
  [SymmetryReduceBZ.jl](https://github.com/jerjorg/SymmetryReduceBZ.jl))

### More ideas
* Support for integrands of the form ``f_{0} ( \int dx_{1} f_{1}( \int dx_{2}
  f_{2}( \cdots \int dx_{n} f_{n})))``
* Multi-dimensional adaptive Chebyshev interpolation, like
  [baobzi](https://github.com/flatironinstitute/baobzi)
* Globally adaptive IAI

## Notes

If you are an interested Python user, see the [Python](@ref) section

To see a poster showcasing calculations with the library, click this
[link](https://web.mit.edu/lxvm/www/slides/Lorenzo_VanMunoz_CCQ_intern_poster_2022.pdf)

## Contact the developer

[Lorenzo Van Muñoz](https://web.mit.edu/lxvm/www/)

[^1]: [Kaye et al. "Automatic, high-order, and adaptive algorithms for Brillouin zone integration"](http://arxiv.org/abs/2211.12959)
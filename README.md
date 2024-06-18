# AutoBZ

| Documentation | Build Status | Coverage |
| :-: | :-: | :-: |
| [![][docs-stable-img]][docs-stable-url] | [![][action-img]][action-url] | [![][codecov-img]][codecov-url] |
| [![][docs-dev-img]][docs-dev-url] |  | [![][aqua-img]][aqua-url] |

`AutoBZ` is an applications library for Brillouin zone integration, including
integrators for density of states, electron density, and optical conductivity.
The methods it uses are high-order accurate, adaptive and well-suited to Wannier
interpolation. Thus, they can reliably resolve sub-meV scale spectral features
with reasonable compute times. For further information and tutorials, please see
[the documentation](https://lxvm.github.io/AutoBZ.jl/stable/).

## Research and citation

If you use AutoBZ.jl in your software or published research works, please
cite one, or, all of the following. Citations help to encourage the development
and maintenance of open-source scientific software.
- This repository: https://github.com/lxvm/AutoBZ.jl
- Our preprint on optical conductivity integration: High-order and adaptive
  optical conductivity calculations using Wannier interpolation. Lorenzo Van
  Muñoz, Jason Kaye, Alex Barnett and Sophie Beck.


## Author and Copyright

AutoBZ.jl was written by [Lorenzo Van Muñoz](https://web.mit.edu/lxvm/www/),
and is free/open-source software under the MIT license.


## Related packages
- [AutoBZCore.jl](https://github.com/lxvm/AutoBZCore.jl)
- [HChebInterp.jl](https://github.com/lxvm/HChebInterp.jl)
- [SymmetryReduceBZ.jl](https://github.com/jerjorg/SymmetryReduceBZ.jl)


<!-- badges -->

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://lxvm.github.io/AutoBZ.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://lxvm.github.io/AutoBZ.jl/dev/

[action-img]: https://github.com/lxvm/AutoBZ.jl/actions/workflows/CI.yml/badge.svg?branch=main
[action-url]: https://github.com/lxvm/AutoBZ.jl/actions/?query=workflow:CI

[codecov-img]: https://codecov.io/github/lxvm/AutoBZ.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/github/lxvm/AutoBZ.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

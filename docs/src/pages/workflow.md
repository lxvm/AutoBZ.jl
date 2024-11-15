# Workflow

## Installation

To install Julia, visit <https://julialang.org/downloads> and use the
recommended method.

`AutoBZ.jl` is an [unregistered Julia
package](https://pkgdocs.julialang.org/v1/managing-packages/#Adding-unregistered-packages),
so it can be added to a Julia environment using the repository URL and
Julia's package manager as follows:
```julia
using Pkg
Pkg.activate()
Pkg.add(url="https://github.com/lxvm/AutoBZ.jl.git", rev="main")
```
The `rev` keyword can also be set to a tagged version, e.g. "v0.5.3".
To download the repository with SSH use the git link:
`git@github.com:lxvm/AutoBZ.jl.git`

The motivation for keeping AutoBZ.jl an unregistered package is that its
intended purpose of calculating specific response functions means that it can be
a dependency of a project, which is reproducible with use of the package
manager, but it is unlikely to become a dependency of another registered
package. Users who would like to develop their own libraries based on this
functionality may use [AutoBZCore.jl](https://github.com/lxvm/AutoBZCore.jl) or
contact the developers with their request.

## Running jobs

Potential users of `AutoBZ` will find scripts in the [`demos`](https://github.com/lxvm/AutoBZ.jl/tree/main/demos) to use as templates
for calculations and jobs. From the project environment configured above, the scripts from
the `demos` folder can be copied with,
```bash
julia --project -e 'using AutoBZ; files = ["svo.wout", "svo_hr.dat", "svo_r.dat", "kc_benchmark.jl"]; cp.(joinpath.(dirname(dirname(pathof(AutoBZ))), "demos", files), joinpath.(pwd(), files))'
```
In the line above, the script `kc_benchmark.jl` was copied along with its default
data files into the working directory. Note that any script
dependencies other than `AutoBZ` will need to be added to the project
environment, e.g. `Pkg.add(["HDF5", "SymmetryReduceBZ"])`.

Finally, to run the script from the command line use the following:
```bash
julia --project kc_benchmark.jl
```
This process is similar to running jobs on a cluster, where it will be benefical
to set the `--threads` flag.

## Contributing

Read the [contributing guidelines](https://github.com/lxvm/AutoBZ.jl/tree/main/CONTRIBUTING.md) for guidance on PRs, issues, and discussions on matters relating to contributing to AutoBZ.jl.

## Development

Anyone interested in developing AutoBZ.jl will likely benefit from a
[Revise.jl](https://timholy.github.io/Revise.jl/stable/) compatible workflow.
The following workflow sets up an environment that makes it convenient to
interactively edit the `AutoBZ` source code and use git to contribute changes
```bash
mkdir dev
git clone https://github.com/lxvm/AutoBZ.jl.git dev/AutoBZ
julia -e 'import Pkg; Pkg.activate("."); Pkg.develop(path="dev/AutoBZ")'
```
Updating the repository can now be done with `cd dev/AutoBZ` and `git pull`.

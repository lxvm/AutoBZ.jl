# AutoBZ.jl demos

These demos are intended to facilitate reproducing calculations and validations
that are intended applications of AutoBZ as well as setting them up for new
materials.

Continue reading for instructions on running the demos.

## First-time setup

To setup Julia, first install either the latest binary from
https://julialang.org/downloads and add it to your path or use your system
package manager to obtain the most recent release. To setup this Julia
environment, `cd` into this repository and execute the line below on your shell.

```
$ julia --project=. -e 'import Pkg; Pkg.develop("..")'
```

Running certain scripts may require additional dependencies, which can be added
with a subsequent command, e.g. `julia --project=. -e 'import Pkg; Pkg.add("SymmetryReduceBZ")'

## Running scripts

To run the scripts in this repository on the Strontium Vanadate example, execute
them from your shell using the format in the line below

```
$ julia --project=. <filename>
```

Once the program exits, the outputs of the script (such as PNG, CSV, or HDF5
files) will be stored in this directory.

If you would like to change the example material to use your own Wannier90
Hamiltonian, you may either replace the "epsilon_mn.h5" file or change the
script to load your file from a different path.

## Developing/editing scripts or `Demos.jl`

To develop, edit, or extend the scripts or `Demos.jl`, I recommend running Julia
interactively, making changes to the source code in your preferred text editor,
and running `include(filename)` every time you want to rerun the code, such as
below

```
$ julia --project=.
julia> include("Demos.jl")
julia> edit("Demos.jl")
julia> include("Demos.jl")
```
# Workflow

Users of `AutoBZ` who would like to use the calculation scripts found in the
`demos` folder of the repository can use the following workflow to run their
jobs either locally or on a cluster. First, create a working directory for this
project that will also serve as the home of its Julia environment
```
$ mkdir workdir
$ cd workdir
$ julia --project=. # start an interactive session in the <workdir> environment
```
In the Julia session, adding `AutoBZ` to the environment while cloning the
repository to the working directory can be done interactively as shown below
```julia
julia> ] # enters the package manager
(workdir) pkg> develop --local https://github.com/lxvm/AutoBZ.jl.git # clones repo into ./dev/AutoBZ
```
From the working directory, the scripts from the `demos` folder can be copied
back, as well any coefficient files for the calculations that will be run.
```julia
julia> ; # enters a shell
shell> cp -t . dev/AutoBZ/demos/svo_hr.dat dev/AutoBZ/demos/svo_r.dat dev/AutoBZ/demos/OC_berry.jl
```
In the lines above, the script `OC_berry.jl` was copied along with its default
data files into the working directory. Note if you have data files with
different names, the script needs to be edited to load those files instead.
Moreover, the files in the working directory can be freely edited without
affecting those in the `AutoBZ` repository, and likewise the repository can be
updated without affecting the files in `workdir` Finally, to run the script from
the Julia session in the `workdir`, simply `include("OC_berry.jl")`. If you
would like to run a script as a cluster job, include the line below in your bash
script submitted to the scheduler.
```
julia --project=. OC_berry.jl
```
When `AutoBZ` is released as a registered package, it will be enough to `add
AutoBZ` to the Julia environment and copy the scripts from the installation
location, although the method described above conveniently saves the repository
in the working directory, letting you easily update the git repository or change
branches. If working in an interactive session, this method also works with a
[Revise.jl](https://timholy.github.io/Revise.jl/stable/) workflow.
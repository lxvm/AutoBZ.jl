# Interfaces

## Wannier90

```@docs
AutoBZ.load_wannier90_data
AutoBZ.load_interp
AutoBZ.load_autobz
```

## PythTB

```@docs
AutoBZ.load_pythtb_data
```

## Python

Julia code, including `AutoBZ.jl`, can be called from Python using the package
[PyJulia](https://pyjulia.readthedocs.io)

### Setup

TL;DR
```
$ julia -e 'import Pkg; Pkg.add("PyCall")'
$ python3 -m pip install julia
```

If you want to, you can install PyJulia in a Python venv, but on the Julia side PyCall.jl must be installed in the default environment

### Demo

The Python snippet below shows how from the `demos` folder of the AutoBZ repository you can run one of the demo scripts:
```
from julia.api import Julia
jl = Julia(compiled_modules=False)

# julia environment setup in working directory 'demos'
jl.eval("""
import Pkg
Pkg.activate(".")
Pkg.instantiate()
""")

# capture output of script
out = jl.eval('include("dos_test.jl")')
```
The first two lines are adapted for loading PyJulia on Debian systems.

## MATLAB

Julia code, including `AutoBZ.jl`, can be called from MATLAB using the package
[MATDaemon.jl](https://github.com/jondeuce/MATDaemon.jl).

### Setup

1. Download the
[`jlcall.m`](https://github.com/jondeuce/MATDaemon.jl/raw/master/api/jlcall.m)
script, which will install the Julia server when first called.
2. Install [Julia](https://julialang.org/) and give MATLAB the path the Julia
   binary by running `setenv('PATH',['path-to-julia/bin:',getenv('PATH')]);`
3. Test that `jlcall` works by running `jlcall('sort', {rand(2,5)},
   struct('dims', int64(2)))`
4. Create setup.jl with the lines `import Pkg; Pkg.activate(".");
   Pkg.develop(path=expanduser("path-to-AutoBZ.jl"));`
5. Start a julia server within MATLAB with the appropriate modules: `jlcall('',
   'project', 'path-to-myproject', 'setup', 'path-to-setup.jl', 'modules',
   {'LinearAlgebra','AutoBZ'}, 'threads', 'auto', 'restart', true);`
6. Now AutoBZ can be used via `jlcall`

### Demo

Suppose we would like to run the function `get_dos` defined in this `script.jl`
```
import Pkg; Pkg.activate("."); Pkg.develop(path=expanduser("path-to-AutoBZ.jl"))
using AutoBZ

function get_dos(seedname, self_energy_path, ωs, rtol, atol)
    H, FBZ = load_wannier90_data(seedname)
    Σ = load_self_energy(self_energy_path)
    integrand = DOSSolver(Σ, H, FBZ, IAI(); ω=first(ωs))
    solver = IntegralSolver(integrand, FBZ, abstol=atol, reltol=rtol)
    return map(ωs) do ω
        update_gloc!(solver; ω)
        solve!(solver).value
    end
end
```
The MATLAB snippet below shows how to setup the Julia server to run a demo
script.
```
jlcall('', ...
    'project', '/path/to/MyProject', % use Julia environment with AutoBZ
    'setup', '/path/to/setup.jl', % path to script shown above
    'modules', {'AutoBZ'}, % import AutoBZ and other modules
    'threads', 'auto', % use the default number of Julia threads
    'restart', true % start a fresh Julia server environment
)
```
and the snippet below shows how to call `get_dos` from MATLAB
```
jlcall('get_dos', '.../svo', '.../svo_self_energy.txt', {0.5})
```
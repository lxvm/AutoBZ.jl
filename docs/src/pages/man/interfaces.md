# Interfaces

## Wannier90

```@docs
AutoBZ.load_wannier90_data
AutoBZ.load_hamiltonian
AutoBZ.load_hamiltonian_velocities
AutoBZ.load_position_operator
AutoBZ.parse_hamiltonian
AutoBZ.parse_position_operator
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
out = jl.eval('include("DOS_test.jl")')
```
The first two lines are adapted for loading PyJulia on Debian systems.

## MATLAB

Julia code, including `AutoBZ.jl`, can be called from MATLAB using the package
[MATDaemon.jl](https://github.com/jondeuce/MATDaemon.jl).

### Setup

Download the
[`jlcall.m`](https://github.com/jondeuce/MATDaemon.jl/raw/master/api/jlcall.m)
script, which will install the Julia server when first called.

### Demo

Suppose we would like to run the function `get_dos` defined in this `script.jl`
```
using AutoBZ

function get_dos(seedname, self_energy_path, ωs, rtol, atol)
    H, FBZ = load_wannier90_data(seedname)
    Σ = load_self_energy(self_energy_path)
    results = run_dos(H, Σ, ωs, FBZ, rtol, atol)
    return results.I
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
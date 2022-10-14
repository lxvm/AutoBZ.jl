# Interfaces

## Wannier90

```@docs
AutoBZ.Applications.parse_hamiltonian
AutoBZ.Applications.load_hamiltonian
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
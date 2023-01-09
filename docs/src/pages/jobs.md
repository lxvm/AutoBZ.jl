# Jobs

The module `AutoBZ.Jobs` has batch functions that setup calculations for
`WannierIntegrand`s, `DOSIntegrand`s, and `KineticIntegrand`s.

## Parallelized scripts

All of the following scripts have parallelization over parameter points
(typically frequencies) enabled by default, except where noted. They also always
use ``k`` point parallelization of equispace integrals.

!!! note "Chemical potential"

    This library does not handle the chemical potential or zero point energy, 
    but recommends that the user offset their Hamiltonian and self-energy data
    by the chemical potential so that the Fermi level is at zero energy. Doing
    so, i.e. with [`AutoBZ.shift!`](@ref), will ensure a consistent convention
    and correctness of the kinetic coefficient calculations, which truncate an
    infinite integral assuming that the Fermi level is at zero energy.

### Recommended

The scripts below are recommended because they return a result from adaptive
integration, but when `rtol>0` they also do an automatic equispace integral as a
precomputation to give appropriate absolute tolerances to the adaptive
integrator, which performs better with absolute tolerances.
```@docs
AutoBZ.Jobs.run_wannier
AutoBZ.Jobs.run_dos
AutoBZ.Jobs.run_kinetic
```

### Adaptive

```@docs
AutoBZ.Jobs.run_wannier_adaptive
AutoBZ.Jobs.run_dos_adaptive
AutoBZ.Jobs.run_kinetic_adaptive
```

### Equispace

The caller supplies a fixed number of ``k`` points per dimension, `npt`, and
should check that the solution is converged with respect to this parameter. In
general, these routines are useful to get fast results with small numbers of
``k`` points (~100) at early stages of a project.
```@docs
AutoBZ.Jobs.run_wannier_equispace
AutoBZ.Jobs.run_dos_equispace
AutoBZ.Jobs.run_kinetic_equispace
```

### Automatic equispace

Frequency parallelization is not enabled by default due to the likelihood of
running out of memory large ``k`` point grids.
```@docs
AutoBZ.Jobs.run_wannier_auto_equispace
AutoBZ.Jobs.run_dos_auto_equispace
AutoBZ.Jobs.run_kinetic_auto_equispace
```

## I/O

```@docs
AutoBZ.Jobs.read_h5_to_nt
AutoBZ.Jobs.write_nt_to_h5
```

## Internal

```@docs
AutoBZ.Jobs.get_safe_freq_limits
AutoBZ.Jobs.batch_smooth_param
```
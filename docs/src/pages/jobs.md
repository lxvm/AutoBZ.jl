# Jobs

The module `AutoBZ.Jobs` has batch functions that setup kinetic coefficient
calculations.

## Scripts

```@docs
AutoBZ.Jobs.run_kinetic
AutoBZ.Jobs.run_kinetic_equispace
AutoBZ.Jobs.run_kinetic_auto
AutoBZ.Jobs.run_kinetic_auto_equispace
```

## Parallelized scripts

```@docs
AutoBZ.Jobs.run_kinetic_parallel
AutoBZ.Jobs.run_kinetic_equispace_parallel
AutoBZ.Jobs.run_kinetic_auto_parallel
AutoBZ.Jobs.run_kinetic_auto_equispace_parallel
```

## I/O

```@docs
AutoBZ.Jobs.read_h5_to_nt
AutoBZ.Jobs.write_nt_to_h5
AutoBZ.Jobs.import_self_energy
```

## Internal

```@docs
AutoBZ.Jobs.get_safe_freq_limits
AutoBZ.Jobs.batch_smooth_param
```
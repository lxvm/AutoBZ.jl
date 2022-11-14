# Jobs

The module `AutoBZ.Jobs` has batch functions that setup optical conductivity
calculations.

## Scripts

```@docs
AutoBZ.Jobs.OCscript
AutoBZ.Jobs.OCscript_equispace
AutoBZ.Jobs.OCscript_auto
AutoBZ.Jobs.OCscript_auto_equispace
```

## Parallelized scripts

```@docs
AutoBZ.Jobs.OCscript_parallel
AutoBZ.Jobs.OCscript_equispace_parallel
AutoBZ.Jobs.OCscript_auto_parallel
AutoBZ.Jobs.OCscript_auto_equispace_parallel
```

## I/O

Note: currently Julia's HDF5 module handles arrays of complex values
inconsistently with Python's HDF5 module. This could be fixed by simply using
PyCall to use Python's `h5py` library instead.

```@docs
AutoBZ.Jobs.read_h5_to_nt
AutoBZ.Jobs.write_nt_to_h5
AutoBZ.Jobs.import_self_energy
```

## Internal

```@docs
AutoBZ.Jobs.BandEnergyBerryVelocities
AutoBZ.Jobs.get_safe_freq_limits
AutoBZ.Jobs.batch_smooth_param
```
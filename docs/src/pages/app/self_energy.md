# Self energies

This section of the documentation explains how scalar and matrix-valued self
energy data are formatted, loaded, and evaluated

## IO

For self energy data stored on equispaced frequency grids, the following file
formats and routines use the `EquiBaryInterp.jl` package to interpolate the
data for continuous evaluation. Otherwise, a rational approximation of the self
energy data is constructed with the AAA algorithm in `BaryRational.jl` and that
is converted into a piecewise Chebyshev interpolant with `HChebInterp.jl`.

### File format

Depending on the amount of data necessary to represent the self-energy, there
are different data file formats for scalar and matrix-valued self energies.
Within matrix-valued self energies. There is also a special case diagonal
matrices.

#### Scalar

First line contains only `nfpts`, the number of frequency points in the data
set. The following `nfpts` lines contain three columns with the frequency point,
real part of the self energy, and imaginary part of the self energy. For example

```
3001
-15.00   7.9224534011421888  -0.0608455837453997
-14.99   7.9107083143558103  -0.0627222170930403
...
```

#### Diagonal

First line contains only `nfpts`, the number of frequency points in the data
set. Second line contains only `num_wann`, the number of bands used for Wannier
interpolation (i.e. should be the same as the Hamiltonian). The following
`nfpts*num_wann` lines contain four columns with the frequency point, one-base
index of the self energy in the diagonal of the matrix, real part of the self
energy, and imaginary part of the self energy. It is assumed that the frequency
data is sorted wih the index as the faster index than the frequency. For example

```
3001
3
-15.00   1   7.9224534011421888  -0.0608455837453997
-15.00   2   7.9224534011422065  -0.0608455837453997
...
```

#### Matrix

First line contains only `nfpts`, the number of frequency points in the data
set. Second line contains only `num_wann`, the number of bands used for Wannier
interpolation (i.e. should be the same as the Hamiltonian). The following
`nfpts*num_wann^2` lines contain five columns with the frequency point, one-base
index of the self energy in the row of the matrix, one-base index of the self
energy in the column of the matrix, real part of the self energy, and imaginary
part of the self energy. It is assumed that the frequency data is sorted wih the
index as the faster index than the frequency. For example

```
3001
3
-15.00   1   1   7.9224534011421888  -0.0608455837453997
-15.00   1   2   7.9224534011422065  -0.0608455837453997
...
```

### Routines

```@docs
AutoBZ.load_self_energy
```

## Interface

```@docs
AutoBZ.AbstractSelfEnergy
AutoBZ.lb
AutoBZ.ub
```

## Types 

```@docs
AutoBZ.EtaSelfEnergy
AutoBZ.ConstScalarSelfEnergy
AutoBZ.ScalarSelfEnergy
AutoBZ.DiagonalSelfEnergy
AutoBZ.MatrixSelfEnergy
```
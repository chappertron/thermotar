# Changelog


## Version 0.0.2

### New Features

- (BREAKING) Can now specify suffix to error columns in `thermo.estimate_error` independently of `error_calc`, defaulting to 'err'.
  Results in the default suffix changing to '_err'
- `__getitem__` is now implemented for `Chunk`
- Added the `OrthogonalBox` type and  a parser for reading it from LAMMPS datafiles. Currently only supports orthogonal boxes

### Breaking

- Suffix to error columns in `thermo.estimate_error` is changed from '_sem' to '_err'.

## Version 0.0.1

Started a changelog

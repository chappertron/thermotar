# The `Thermo` Class

The `Thermo` class is used to load [LAMMPS log files](https://docs.lammps.org/Run_output.html),
and to a lesser extent, GROMACS `.xvg` files.

The class is a thin wrapper of a Pandas `DataFrame`. 
The underlying DataFrame can be accessed with the `.data` method.

The primary way to create instances of this class is using the [`create_thermos`][thermotar.thermo.Thermo.create_thermos]
class method. 
This constructor takes in a path for the `LAMMPS` logfile and returns either one 
or a list of `Thermo` objects, depending on whether the `join` kwarg is set.

As well as parsing, the `Thermo` class has several methods that assist with analysis of 
simulation results.


::: thermotar.thermo
    handler: python

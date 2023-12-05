
# The Thermo Class
The thermo class is used to load lammps log files, and to a lesser extent, gromacs `.xvg` files

The primary way to create these is using `thermotar.create_thermos`, which takes in a path for the `LAMMPS` logfile and 
returns either one or multiple thermo objects, depending on whether `join` and `last` are set or not.

::: thermotar.thermo

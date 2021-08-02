# thermotar
For importing and generating some LAMMPS and GROMACS files types, with a light bit of analysing too

This is very alpha software. Well whatever comes before alpha really.


Installation:

  Currently no pip or conda distribution.
  
  Easiest way to install is to run setup.py in thermotar.
  
  Specify the develop option when doing so, so that changes you make are reflected in the installation.
  
  
  
  
  
Basic usage:


To read a lammps log file:
  import thermotar as th
  thermo = th.create_thermos('log.lammps')
  
The thermo object then contains a .data method which is a pandas dataframe containing all the data from the last set of log values.







  

# thermotar
For importing some LAMMPS and GROMACS files types, with a light bit of analysing too.

This is very alpha software. Well whatever comes before alpha really.


## Installation:
  Quick install:
  Run from the root directory of the project:
  ```sh
  pip install .
  ```

  Currently no pip or conda distribution.
  
  <!-- Easiest way to install is to run setup.py in thermotar. -->
  
  Specify the develop option when doing so, so that changes you make are reflected in the installation.
  
  
  
  
  
## Basic usage
To read a lammps log file:
``` python
  import thermotar as th
  # Read LAMMPS log file
  # By default reads the last runs output
  thermo = th.create_thermos('log.lammps')

  import matplotlib.pyplot as plt
  plt.plot(thermo.Step,thermo.Temp)
  plt.show()

  # Read data from chunk ave or RDF
  # Reads only the last table
  chunk = th.create_chunk('chunk.prof')
  plt.plot(chunk.Coord1,chunk.temp)
  plt.show()

  # Read all the chunks from a chunk/ave file
  chunks = th.create_multi_chunks('chunk.prof')
  # Average the different frames of multichunk into a single chunk
  chunk = chunks.flatten_chunk()
``` 
The thermo object then contains a .data method which is a Pandas dataframe containing all the data from the last set of log values.







  

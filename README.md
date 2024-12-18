# thermotar
For importing LAMMPS log and chunk files, with a light bit of analysing too.

Docs are located at: [chappertron.github.io/thermotar/](https://chappertron.github.io/thermotar/)


## Installation:
  Quick install:
  Clone the repository and run from the root directory of the project:
  ```sh
  pip install .
  ```

  Alternatively build without explicitly cloning first:
  ```sh
  pip install git+https://github.com/chappertron/thermotar
  ```

  Currently no pip or conda distribution. 
  
  
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







  

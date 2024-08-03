# Welcome to Thermotar

## Quickstart

### Installation
Run from the root directory of the project:
```bash
pip install .
```
To read a lammps log file:
``` python
  import thermotar as th
  # Read LAMMPS log file
  # By default reads the last runs output
  thermo = th.create_thermos('log.lammps')

  # Read data from chunk ave or RDF
  # Reads only the last table
  chunk = th.create_chunk('chunk.prof')
``` 


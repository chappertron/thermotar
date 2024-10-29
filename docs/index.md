# Welcome to Thermotar

## Quickstart

### Installation
Run from the root directory of the project:
```bash
pip install .
```
Alternatively build without explicitly cloning first:
```sh
pip install git+https://github.com/chappertron/thermotar
```

Currently, there is no pip or conda distribution. 
  


### Examples
```python
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


## Full Documentation
The reference section contains documentation for the main package.

Useful places to look are:
- [`thermo`](reference/thermo.md)
- [`chunk`](reference/chunk.md)
- [`interface`](reference/interface.md)

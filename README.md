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
### Reading a lammps log file:
``` python
  import thermotar as th
  # Read LAMMPS log file
  # By default reads the last runs output
  thermo = th.create_thermos('log.lammps')

  import matplotlib.pyplot as plt
  plt.plot(thermo.Step,thermo.Temp)
  plt.show()

 
``` 
The columns of data can be accessed from the `Thermo` object using indexing or the accessor syntax,
```python
temp = thermo['Temp']
temp = thermo.Step
```
The latter can only be used if the columns are valid Python identifiers.
To aid with this, the `create_thermos` function tidies up the column names.

Alternatively, the `Thermo` object has a `.data` attribute which contains the underlying Pandas `DataFrame`.
Then any normal Pandas operations can be used on it.

By default only the last 'run' from the log file is loaded with `th.create_thermos`.
Using the keyword arguments `last=False` and `join=False` will create a list of `Thermo` objects, one for each run. 
Using `last=False` and `join=True` will create a single `Thermo` object, with all the runs in a single file.
`last=True` and `join=True` are the defaults.

### Reading 'Chunk' Files

Chunks are spatial data outputted from the LAMMPS `fix chunk/ave`.
This file contains multiple tables, each of which is a sub-average over the course of the simulation.

If the `ave running` argument is provided to `fix chunk/ave`, then only the last frame is needed.
The `th.create_chunk` function reads just the last frame of this file.

``` python
  # Read data from chunk ave or RDF
  # Reads only the last table
  chunk = th.create_chunk('chunk.prof')
  plt.plot(chunk.Coord1,chunk.temp)
  plt.show()
```

If you need all the frames, then the `th.create_multi_chunks` function can create a `MultiChunk` object to load them all.

``` python
  # Read all the chunks from a chunk/ave file
  chunks = th.create_multi_chunks('chunk.prof')
  # Average the different frames of `MultiChunk` into a single `Chunk`
  chunk = chunks.flatten_chunk()
```

As with the `Thermo` class, `Chunk` and `MultiChunk` instances have a `.data` attribute, and can be indexed or use the getter syntax.

The `Chunk` format is also outputted by the LAMMPS command `fix ave/time` in vector mode.

The constructor `th.create_chunk` can also optionally read data outputted by the `numpy` function `savetxt` using the `style='np'` kwarg,
``` python
  chunk = th.create_chunk('data.txt',style='np')
```






  

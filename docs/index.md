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

<!-- For full documentation visit [mkdocs.org](https://www.mkdocs.org). -->

<!-- ## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

## Test addition -->

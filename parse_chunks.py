import re
from io import StringIO
import pandas as pd


def _lammps_parse_chunk(chunk_file, nchunks='auto', header_rows = 3,last_only = False ):
    '''
    
    Parser for the types of files outputed by compute ave/correlate and compute chunk commands
    
    these typically have a file format where there are many tables in sets of timeseries

    For convinence this parsing gives the output as pandas objects, with indexes as the 

    The typical format of a lammps 'chunk' dump is:

    3 row header:
    title: # Chunk averaged... 
    column labels for each table 'index' # Timestep Number-of-chunks Total-count
    columns of labels each variable outputted: # Chunk Coord1 Ncount v_temp_atom ... 

    Then there is:

    an 'index' Timestep N-of-chunks Total-count - occurs every Number-of-chunks lines
    a table of chunked data, with Number-of-chunks rows



    nchunks = 'auto' or int
    last_only: bool, default false. if true, will only use the last nchunks lines




    '''

    # find header lines
    # find number of columns for each index and in each sub table
    if nchunks == 'auto':
        for n,line in enumerate(chunk_file):
            if n == header_rows:
                #get N-of-Chunks from the first instance of the 'index', which is after the head
                nchunks = float(line.split()[1])
    
    

    # for line in chunk_file:
    #     # append to indexes if of the length indicies 
    #     if line.startswith('#'):
    #         # skip commented lines
    #         next


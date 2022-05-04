from dataclasses import dataclass
from pathlib import Path
from typing import Type
import pandas as pd
from . import df_utils
import numpy as np
from collections import OrderedDict

from dataclasses import dataclass

# @dataclass
class LMPChunksParser:
    def __init__(self,fname,header_rows = 3):
        self.header_line = header_rows -1
        self.fname:str = fname
        self.column_names:list = self.set_columns()
        self.data : pd.DataFrame = pd.DataFrame()

    def set_columns(self,):
        return df_utils.comment_header(self.fname,line_no=self.header_line)

    def get_info_rows(self,verbose = False):
        start_index = 3
        next_info_row = start_index
        self.info_rows_dicts = OrderedDict()
        with open(self.fname) as stream:
            for i, line in enumerate(stream.readlines()):
                # print(f"{i} {line}")
                if i == next_info_row:
                    parsed_line = line.split() # split line with whitespace
                    try:
                        parsed_line = [float(x) for x in parsed_line]
                        parsed_line[1] = int(parsed_line[1])
                        parsed_line = tuple(parsed_line)
                    except TypeError as e:
                        raise e
                    if verbose: print(f"{i}{parsed_line}")
                    next_info_row += parsed_line[1] + 1 ## skip to the next "info" line 
                    self.info_rows_dicts[i] = parsed_line

    def parse_chunks(self):
        # print(self.info_rows_dicts)

        # pass through the file a second time to create the dict of arrays.
        # line_nos = list(info_rows_dicts.keys())
        values = {}
        for i, (line_no, info)  in enumerate(self.info_rows_dicts.items()):
            # next_line_no = line_nos[i+1]
            # print(line_no,int(info[1]))
            values[info] = pd.DataFrame(np.loadtxt(self.fname,skiprows=line_no+1,max_rows=int(info[1])),columns=self.column_names)
        
        self.data = pd.concat(values)

    def set_index_name(self):
        ind_names = tuple(df_utils.comment_header(self.fname,line_no=self.header_line-1))
        # print(ind_names)
        self.data.index.name = ind_names

def parse_lmp_chunks(chunk_file,header_rows = 3):
    parser = LMPChunksParser(chunk_file,header_rows=header_rows)
    parser.get_info_rows()
    parser.parse_chunks()
    parser.set_index_name()
    return parser

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


    header_line : int or None  if none, the header is loaded from last line of header rows, else it is the index specified

    '''

    

    pass

    # # TODO add option for no header
    # if not header_line:
    #     # if no header line is specified, assume second of the header rows
    #     header_line = header_rows - 1
    # header = df_utils.comment_header(fname,line_no=header_line)

    
    # #find N automactically
    # if nchunks =='auto':
    #     # TODO implement automatically finding this!
    #     # read the NHeader+1 th lines indexing starts at zero tho
        
    #     with open(fname) as stream:
    #         for i,line in enumerate(stream):
    #             if i == header_rows:
    #                 # pick the second thing as index NB. currently assumes constant number of bins
    #                 N = int(line.split()[1])
    #                 break
    # else:
    #     N = nchunks
    # last_N_IO = tail(fname,N=N) #StringIO object

    # return pd.read_csv(last_N_IO,sep=r'\s+',names=header)
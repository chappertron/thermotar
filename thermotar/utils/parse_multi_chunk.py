from dataclasses import dataclass
from pathlib import Path
from typing import Type
import pandas as pd
from . import df_utils
import numpy as np
from collections import OrderedDict

# @dataclass
class LMPChunksParser:
    __slots__ = 'header_line', 'fname','column_names','data','info_rows_dicts','info_row_indicies','t_steps','n_chunks','total_counts'
    def __init__(self,fname,header_rows = 3,):
        self.header_line = header_rows -1
        self.fname:str = fname
        self.column_names:list = self.set_columns()
        self.data : pd.DataFrame = pd.DataFrame()
        self.info_rows_dicts : OrderedDict = OrderedDict()
        self.info_row_indicies = []  # containing the info found from 
        self.t_steps = []
        self.n_chunks = []
        self.total_counts = []

    def set_columns(self,):
        return df_utils.comment_header(self.fname,line_no=self.header_line)

    def get_info_rows(self,rerun = False,verbose = False):
        '''
            Creates a dictionary of row indicies and parsed lines 
        
        '''
        if len(self.info_row_indicies) != 0 and not rerun :
            raise ValueError('Already found indicies, to rerun pass the rerun option')
        start_index = self.header_line+1
        next_info_row = start_index
        # self.info_rows_dicts = OrderedDict()
        with open(self.fname) as stream:
            for i, line in enumerate(stream.readlines()): # Lazily read
                # print(f"{i} {line}")
                if i == next_info_row:
                    parsed_line = line.split() # split line with whitespace
                    try:
                        parsed_line = [float(x) for x in parsed_line]
                        parsed_line[1] = int(parsed_line[1])
                        parsed_line = tuple(parsed_line)
                    except TypeError as e:
                        raise e
                    if verbose>2: print(f"{i}{parsed_line}")
                    next_info_row += parsed_line[1] + 1 ## skip to the next "info" line 
                    # self.info_rows_dicts[i] = parsed_line
                    self.update_chunk_locs(i,parsed_line)

    def update_chunk_locs(self,i:int,parsed_line:tuple):
        self.info_row_indicies.append(i)
        self.t_steps.append(parsed_line[0])
        self.n_chunks.append(parsed_line[1])
        self.total_counts.append(parsed_line[2])
        self.info_rows_dicts[i] = parsed_line


    def parse_chunks(self,chunk_len ='auto',*,verbose = False ):
        '''
            TODO rather than looping through the whole file in the first place, give the option to assume that all chunks are the same size.
            First checks if all the rows are the same length
            TODO add chunk_len = "equal" that uses the first chunk size found and then assumes that the rest of the chunks are the same size 
            TODO add chunk_len : int = n , manually set the chunk size to n
        
        '''
        if chunk_len != 'auto':
            raise NotImplementedError('Only the auto method is currently implemented for finding the chunks')

        if chunk_len == 'auto' and len(self.info_row_indicies) == 0:
            raise ValueError("For chunk_len='auto', run get_info_rows method first!")

        use_single_size = False
        arr_n_chunks = np.array(self.n_chunks)
        if chunk_len=='auto' and arr_n_chunks.min() ==arr_n_chunks.max():
            if verbose: print('Using the single chunk size parser')
            use_single_size = True 
            chunk_size = arr_n_chunks[0]
        else:
            if verbose: print('Using the multisize chunk size parser')
            use_single_size=False

        if use_single_size:
            self.parse_chunks_single_size(chunk_size = chunk_size)
        else:
            self.parse_chunks_multi_size(method='np')  ## passed through as kwargs



    def parse_chunks_single_size(self,chunk_size,*,verbose = False): 
        '''
            TODO make the index just the timestep, and not the chunk size,etc 
            TODO allow the system to be totally lazy,
            TODO parrellelize iteration over the chunks?
        '''
        if len(self.info_row_indicies) > 0 :
            # if these rows have been found, they can be skipped.
            skiprows = self.info_row_indicies
        else: 
            raise NotImplementedError('Not yet added lazy way to skip header rows')
            ### to keep things lazy, just make chunks chunksize+1, then remove the header rows from the top of each chunk
            skiprows = None
        
        values = {}
        with pd.read_csv(self.fname,sep="\s+",chunksize=chunk_size,skiprows=skiprows,names = self.column_names,comment='#') as reader:
            for i,chunk in enumerate(reader):
                ind = (self.t_steps[i],self.n_chunks[i],self.total_counts[i])
                values[ind] = chunk.reset_index(drop=True)
      
        self.data =  pd.concat(values)

        return self.data 


    
    def parse_chunks_multi_size(self,method ='np',**kwargs):
        if method =='np':
            self.parse_chunks_np(**kwargs)
        if method=='pd':
            self.parse_chunks_pd(**kwargs)



    def parse_chunks_np(self):
        # print(self.info_rows_dicts)

        # pass through the file a second time to create the dict of arrays.
        # line_nos = list(info_rows_dicts.keys())
        if len(self.info_row_indicies) == 0:
            raise ValueError("Run get_info_rows method first!")
        values = {}
        for  (line_no, info)  in self.info_rows_dicts.items():
            # next_line_no = line_nos[i+1]
            # print(line_no,int(info[1]))
            values[info] = pd.DataFrame(np.loadtxt(self.fname,skiprows=line_no+1,max_rows=int(info[1])),columns=self.column_names)
        
        self.data = pd.concat(values)

    def parse_chunks_pd(self):
        # print(self.info_rows_dicts)

        # pass through the file a second time to create the dict of arrays.
        # line_nos = list(info_rows_dicts.keys())
        if len(self.info_row_indicies) == 0:
            raise ValueError("Run get_info_rows method first!")
        values = {}
        for i, (line_no, info)  in enumerate(self.info_rows_dicts.items()):
            # next_line_no = line_nos[i+1]
            # print(line_no,int(info[1]))
            values[info] = pd.read_csv(self.fname,skiprows=line_no+1,nrows=int(info[1]),names=self.column_names,sep='\s+',engine='c')
        
        self.data = pd.concat(values)

    def set_index_name(self):
        ind_names = tuple(df_utils.comment_header(self.fname,line_no=self.header_line-1))
        # print(ind_names)
        self.data.index.name = ind_names

def parse_lmp_chunks(chunk_file,header_rows = 3,verbose = False):
    parser = LMPChunksParser(chunk_file,header_rows=header_rows)
    parser.get_info_rows(verbose=verbose)
    if verbose : print("Calling parse_chunks")
    parser.parse_chunks(chunk_len="auto")
    # parser.parse_chunks_np()
    # print("Calling Parse Chunks pd" )
    # parser.parse_chunks_pd()
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
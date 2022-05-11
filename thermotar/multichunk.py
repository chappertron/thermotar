
from ast import Mult
from optparse import NO_DEFAULT
from typing import Iterable
from .chunk import Chunk
from .utils.parse_multi_chunk import LMPChunksParser, parse_lmp_chunks
from .utils import lmp_utils,df_utils
import pandas as pd
import  numpy as np

class MultiChunk():

    def __init__(self, df, file2=None, CLEANUP=True, coord_cols=['Coord1', 'Coord2', 'Coord3','coord'], centred=False,centered=None, **kwargs):

        ''' thermo_file - string of log file location '''
        self.data : pd.DataFrame = df

        # clean up dataframe

        #apply strip_pref function to remove 'c_/f_/v_' prefixes to all columns
        if CLEANUP:
            self.data.rename(columns = lmp_utils.strip_pref, inplace = True)
            self.data.rename(columns=lmp_utils.drop_python_bad, inplace= True)
            #todo merge columns into vectors

        # set the columns as attributes
        for col in self.data.columns:
            #setattr(self, col ,getattr(self.data, col))
            # has to be set to a method of the class
            setattr(self.__class__, col, df_utils.raise_col(self,col)) # set attribute to this property object??
        # column names for the coordinates, up to 3
        # only those in the df are included, by finding intersect of sets.
        self.coord_cols = list(set(self.data.columns.to_list()) & set(coord_cols))
        if centered is not None:
            centred = centered

        self.centred = centred # Initialise assuming asymmetrical - to do implement a method to check this!!!!

    def copy(self):
        new_chunk = MultiChunk(self.data.copy())
        new_chunk.coord_cols = self.coord_cols
        new_chunk.centred = self.centred
        return new_chunk

    def zero_to_nan(self,val = 0.0,col=None):
        '''
            Replace an exact value specified by val with nan
            In columns col
            Improves averages
        '''
        if col is None:
            to_replace = {val:np.nan}
            value = np.nan
            self.data = self.data.replace(to_replace=to_replace)
            return 
            # self.data = self.data.replace(to_replace=) 
        elif isinstance(col,list):
            to_replace = {column:val for column in col }
            value = np.nan
        else:
            to_replace = {col:val}
            value = np.nan    
        self.data = self.data.replace(to_replace=to_replace,value = value)
    def zero_to_nan_return(self,val=0.0,col=None):
        '''
            Replace an exact value specified by val with nan
            In columns col
            Improves averages
            Returns a new MultiChunk
        '''
        new_chunk =self.copy()
        new_chunk.zero_to_nan(val=val,col=col)
        return new_chunk


    def thresh_to_nan_inplace(self,col, thresh = 0.0,):
        '''
            Replace values below a threshold with Nan
            In columns col
            Improves averages

            TODO actually implement what this says, not just removing the rows.....
        '''
        df = self.data
        
        self.data = self.data.loc[df[col] >= thresh ]

    def thersh_to_nan_return(self,col,thresh = 0.0):
        '''
            Replace values below a threshold with Nan
            In columns col(ums)
            Improves averages
            Returns a new object 
        '''
        # TODO use views or something instead, idk
        # TODO create a clone method for chunk/multichunk/ for the the "DataHolder" class...
        new_chunk = MultiChunk(self.data.copy())
        new_chunk.thresh_to_nan_inplace(col=col,thresh = thresh)

        return new_chunk
      

    def zero_to_nan_return(self,val = 0.0,col=None):
        '''
            Replace an exact value specified by val with nan
            In columns col
            Improves averages
            Same as zero_to_nan but creates a new obj
        '''
        if col is None:
            # apply to all cols
            return MultiChunk(self.data.replace(to_replace={val:np.nan}).copy())
        else:
            return MultiChunk(self.data.replace(to_replace={col:val},value = np.nan).copy())
        

    def flatten_chunk(self, drop_na = None) -> Chunk :
        
        df = self.data.groupby(level=(3)).mean()

        if drop_na is not None:
            df.dropna(how=drop_na,inplace=True)

        return Chunk(df)


    @staticmethod
    def create_multi_chunks(fname,*, verbose = False):
        parser = parse_lmp_chunks(fname,verbose = verbose)
        return MultiChunk(parser.data)





if __name__ == "__main__":
    test_chunk = MultiChunk.create_multi_chunks('./tests/test_files/temp.oxytop')
    test_chunk.zero_to_nan()
    test_chunk.flatten_chunk() 
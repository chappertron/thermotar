''' Defines a thermo class
    Thermo Data is extracted from log files
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from io import StringIO
# from .utils import lmp_utils
from .utils import lmp_utils
from .utils import parse_chunks

class Chunk():

    def __init__(self,thermo_df,file2 = None,CLEANUP = True, **kwargs):

        ''' thermo_file - string of log file location '''
        self.data : pd.DataFrame = thermo_df

        # clean up dataframe

        #apply strip_pref function to remove 'c_/f_/v_' prefixes to all columns
        if CLEANUP:
            self.data.rename(columns = lmp_utils.strip_pref, inplace = True)

            #todo merge columns into vectors

        # set the columns as attributes
        for col in self.data.columns:
            setattr(self, col ,getattr(self.data, col))

    @staticmethod
    def create_chunk(fname,last = True, style='lmp'):
        if style == 'lmp':

            if last:
                df = parse_chunks.lmp_last_chunk(fname,nchunks='auto',header_rows=3)
            else:
                raise NotImplementedError('Only the last chunk is supported at the moment.')
        else:
            raise NotImplementedError('Only LAMMPS chunk files supported so far.')

        return Chunk(df)




if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    # testing the chunk creator

    chunk_test = Chunk.create_chunk('../test_files/temp.profile')

    plt.plot(chunk_test.Coord1,chunk_test.data['density/number'])
    plt.show()
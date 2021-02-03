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
from thermotar.utils import lmp_utils
from thermotar.utils import parse_chunks

class Chunk():

    def __init__(self, thermo_df, file2=None, CLEANUP=True, coord_cols=['Coord1', 'Coord2', 'Coord3','coord'], centred=False,centered=None, **kwargs):

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
        # column names for the coordinates, up to 3
        # only those in the df are included, by finding intersect of sets.
        self.coord_cols = list(set(self.data.columns.to_list()) & set(coord_cols))
        if centered is not None:
            centred = centered

        self.centred = centred # Initialise assuming asymmetrical - to do implement a method to check this!!!!

    # Property Definitions

    @property
    def centered(self):
        return self.centred



    @classmethod
    def create_chunk(cls,fname,last = True, style='lmp'):
        # class method used so inhereted classes can also use this method
        if style == 'lmp':

            if last:
                df = parse_chunks.lmp_last_chunk(fname,nchunks='auto',header_rows=3)
            else:
                raise NotImplementedError('Only the last chunk is supported at the moment.')
        elif style == 'np':
            # read the data frame from a space seperated file outputted by np.savetxt. 
            df = parse_chunks.parse_numpy_file(fname,header_row=0, comment = '#')
        else:
            raise NotImplementedError('Only LAMMPS chunk files and numpy save_txt files are supported so far.')

        return cls(df)


    @classmethod
    def create_empty(cls,size = None):
        '''Create an empty chunk/inherited so that the missing data can be handeled more effectively.'''
        #return cls(df)
        raise NotImplementedError('TODO: implement this')
        pass
    




    def centre(self, coord='all',moment=None):
        '''
        Shift the origin of the simulation box to match

        coord: index of coordinate column to centre over, indexes self.coord_cols. default is all

        moment: if Not one, centres the system to this column name.
        
        '''


        if coord=='all':
            # calculate the union of the list of coord_cols and the df columns
            coords = self.coord_cols

        elif type(coord) == int:
            coords = [self.coord_cols[coord]]
        else:
            # if neither a number or 'all', assumes a column name
            coords = [coord]
        # iterate over selected coordinates and perform the centring operation
        for coord_col in coords:
            if moment:
                self.data[coord_col] -= self.moment(coord_col,moment) # set origin to be first moment, weighted by moment parameter
            else:
                self.data[coord_col] = Chunk.centre_series(self.data[coord_col])

        self.centred = True
            
    def center(self,coord='all'):
        '''An alias of centre for yanks'''
        self.centre(self,coord=coord)
        pass

    @staticmethod
    def centre_series(series:pd.Series):
        '''
            Subtract the average of a series from the series.
        '''
        centre = (series.max()+series.min())/2
        centred = series - centre
        return centred
    

    def moment(self, coord, weighting,order = 1, integrate= 'trapz'):
        '''Calculate the specified moment of the coordinate, weighted by a named property'''
        coords = self.data[self.choose_coordinate(coord)].T
        
        integrand = self.data[weighting]*coords**order
        normaliser = np.trapz(self.data[weighting],coords)

        return np.trapz(integrand,coords)/normaliser


    def choose_coordinate(self, coord):
        '''if an integer, indexes the self.coord_cols field,
            if string 'all', returns self.coord_cols
            if any other, returns 
        '''

        if coord=='all':
            # calculate the union of the list of coord_cols and the df columns
            coords = self.coord_cols

        elif type(coord) == int:
            coords = [self.coord_cols[coord]]
        else:
            # if neither a number or 'all', assumes a column name
            coords = [coord]

        return coords
    def fold(self, crease = 0,coord_i=0):
        '''
            Fold the profile about z = 0
            warning: if properties have been calculated prior to folding, they may no longer be correct.

            For example, electric fields calculated by integrating charge profiles, will have a different sign in each part of the box.

            crease: -
                    Position along folded coordinate to fold about

            coord : int -
                    The index of the self.coord_cols list. Default is 0, the first coord





        '''

        coord_name = self.coord_cols[coord_i]

        if (crease == 0) and self.centred:
            # don't bother finding fold line, just go straight to folding!!!
            # in this case fold by making all negative and 
            self.data[coord_name] = np.absolute(self.data[coord_name])

        
        # TODO Implement fully, even when not centred!


        pass












if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    # testing the chunk creator

    chunk_test = Chunk.create_chunk('../test_files/temp.profile')
    print(chunk_test.coord_cols)
    plt.plot(chunk_test.Coord1,chunk_test.data['density/number'])
    plt.show()
    chunk_test.centre()
    plt.plot(chunk_test.Coord1,chunk_test.data['density/number'])

    plt.show()

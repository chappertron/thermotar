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
from .utils import df_utils
from .utils.df_utils import raise_col, raise_columns

class Chunk():

    def __init__(self, thermo_df, file2=None, CLEANUP=True, coord_cols=['Coord1', 'Coord2', 'Coord3','coord'], centred=False,centered=None, **kwargs):

        ''' thermo_file - string of log file location '''
        self.data : pd.DataFrame = thermo_df

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
            setattr(self.__class__, col, raise_col(self,col)) # set attribute to this property object??
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
    def create_chunk(cls,fname,style='lmp',last = True):
        ''' 
            Load LAMMPS or numpy savetxt as a df and then create a Chunk instance, or instance of an inherited class
            TODO: Implement .xvg styles
            TODO: Implement auto choosing the style, either with file extensions or some form of parsing
        '''
        
        
        # class method used so inhereted classes can also use this method
        # if style == 'auto':
        #     try:
        #         with open(fname,'r') as stream:
        #             for i,line in enumerate(stream):
        #                 if i==3: int(line.split()[1]) # try to cast into an integer, if this fails, likely not lmp style
        #         style = 'lmp'
        #     except ValueError:
        #         # set style = 'np'
        #         style = 'np'
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
    def create_empty(cls,df,size = None):
        '''Create an empty chunk/inherited so that the missing data can be handeled more effectively.'''
        return cls(df,CLEANUP=False)
        #raise NotImplementedError('TODO: implement this')
        #pass
    
    def raise_columns(self):
        '''Raise columns from the df to be attributes
            I have no clue how pandas does this automatically...
            Maybe I need to make it so my objects can be indexed 
            and in doing that for assignment, the attributes can be raised up
            TODO : Something with above??
        '''
        df_utils.raise_columns(self)

    def prop_grad(self,prop,coord,**kwargs):
        '''
            creates the gradient of prop w/ respect to coord. better than direct assignment because updates the columns
        '''

        df = self.data

        df[prop+'_grad'] = np.gradient(df[prop],df[coord],**kwargs)

        # updates the columns
        df_utils.raise_columns(self)


    def nearest(self,property,value,each_half = True,coord = 0):

        '''
        Currently returns index

        To do, return the actual row of properties??????

        This way if nan - make a nan list

        '''
        coord = self.data[self.coord_cols[coord]] # select coordinate column for which to find nearsest in each side
        
        prop = self.data[property]
        # if each half, find the index in each half of the simulation box

        left = prop.loc[coord < 0 ]
        right = prop.loc[coord > 0 ]
        
        right_nearest_index =right.sub(value).abs().idxmin()

        if len(left.index) !=0:
            left_nearest_index = left.sub(value).abs().idxmin()
            indexes = [left_nearest_index,right_nearest_index]
        else:
            indexes = [right_nearest_index]

        try:
            return self.data.loc[indexes].copy()
        except KeyError: # just create an empty row
            row_copy = self.data.loc[(0,),:].copy() # janky indexing makes it a df rather than a series
            row_copy.loc[:] = np.nan
            return row_copy


    def centre(self, coord='all',moment=None):
        '''
        Shift the origin of the simulation box to match

        coord: index of coordinate column to centre over, indexes self.coord_cols. default is all

        moment: if Not None, centres the system to this column name, weighted by this column name
        
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
                #print(coord_col)
                self.data[coord_col] = self.centre_series(self.data[coord_col])

        self.centred = True
            
    def center(self,coord='all'):
        '''An alias of centre for yanks'''
        self.centre(coord=coord)
        pass

    @staticmethod
    def centre_series(series:pd.Series):
        '''
            Subtract the average of a series from the series.
        '''
        centre = (series.max()+series.min())/2
        centred = series - centre
        return centred
    
    def parity(self,prop,coord = 0):
        '''
            Multiplies a property by the sign of the coordinate. Should only be applied to properties that are pseudo scalars,  i.e. change sign under coordinate inversion, so that upon folding and averaging
            properties are correct.
        '''

        # centre first 
        if type(coord) == int:
            coord = self.coord_cols[coord]
        if not self.centred:
            self.centre()

        self.data[prop+'_original'] = self.data[prop]
        self.data[prop] *= np.sign(self.data[coord]) # multiply by the sign of the coordinate column
        self.raise_columns()
        

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
    def fold(self, crease = 0,coord_i=0, sort = False, inplace = True):
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
        elif crease == 0:
            # if the crease is located at coord = 0, but not already centred then - centres
            self.centre(coord=coord_name)
            self.data[coord_name] = np.absolute(self.data[coord_name])
        else:
            # folding about some other value
            self.data[coord_name] -= crease # set origin to the value to be creased about
            self.data[coord_name] = np.absolute(self.data[coord_name]) # get absolute value
            self.data[coord_name] += crease # add the crease value back so still starts at this value!

        # TODO Implement fully, even when not centred!

        # TODO auto sort ?

        if sort:
            # sorts the data by this column name
            self.data.sort_values(by = coord_name ,inplace = True,ignore_index = True) # labels 0,1,2.... Needed for future integration/differentaion operations in numpy

        pass


    def rebin(self,coord,bw=0.25,n_bins = None,mode ='average',inplace=False):
        ''' 
            TODO - Change so it implements the dfutils rebin function
            TODO - impelement n_bins argument
            Rebin the data based on coordinates for a given new bin width.
            Default is to perform averages over these bins.
            Could also add weightings for these averages

            coord = Column name of the coordinate to create the bins from

            inplace : bool  
                if True, overwrites the .data method, else just returns the data frame.

        '''

        df = self.data
        
        # coord_max = df[coord].max()
        # coord_min = df[coord].min() # finding min and max so it works with gfolded data

        # n_bins = (coord_max-coord_min) // bw #double divide is floor division
        # n_bins = int(n_bins)

        # bins = pd.cut(df[coord],n_bins)


        # df_grouped = df.groupby(bins)

        
        df_binned = df_utils.rebin(df,coord,bw=bw,mode = mode)
        
        # if mode == 'average' or mode == 'mean':
        #     df_binned = df_grouped.mean()
        # else:
        #     df_binned = df_grouped.sum()

        if inplace:
            self.data = df_binned
        else:
            return df_binned
        


    def fold_and_ave(self,crease = 0, coord_i=0, sort = True,bw='auto'):
        '''
                Fold the profile about z = 0
                warning: if properties have been calculated prior to folding, they may no longer be correct, epsecially properties that invert under coordinate inversion (pseudoscalars)

                For example, electric fields calculated by integrating charge profiles, will have a different sign in each part of the box.

                crease: -
                        Position along folded coordinate to fold about

                coord : int -
                        The index of the self.coord_cols list. Default is 0, the first coord
                        if all, will fold all coordinates, but will only crease

            bw - need to re bin ! if auto - tries to work out the original bin spacing and then groups by this, if not auto, specify the bin width in distance 

        '''
       

        if coord_i == 'all':
            coord_names = self.coord_cols
        else:
            coord_names = [self.coord_cols[coord_i]]

        if crease == 0 and not self.centred:
            # if the crease is located at coord = 0, but not already centred then - centres
            self.centre(coord=coord_names)


        df = self.data.copy()

        coord1 = df[coord_names[0]]

        if bw == 'auto':
            bw = np.abs(coord1.iloc[-1]-coord1.iloc[-2])  # if auto work out from the difference of the last 2 points of the coord



        # # only index by the first coord,but flip all?
        # select_a = (df[coord_names[0]] >=  0)
        # select_b = ~select_a

        # df_a = df.loc[select_a].sort_values(by = coord_names[0] ,inplace = True,ignore_index = True)
        # df_b = df.loc[select_b]
        # df_b[coord_names]=df_b[coord_names].abs()
        # df_b.sort_values(by = coord_names[0] ,inplace = True,ignore_index = True)

        # df_ave = pd.concat({'a':df_a,'b':df_b}).mean(level=1)    
        
        #fold the df
        df[coord_names] = np.abs(df[coord_names]-crease)+crease # assumes already centred for now 
        df_ave = df_utils.rebin(df,coord_names[0],bw).sort_values(coord_names[0]) # performing the rebinning # then sort by the coord
        

        return df_ave



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    # testing the chunk creator

    chunk_test = Chunk.create_chunk('./test_files/temp.profile')
    # print(chunk_test.coord_cols)
    # plt.plot(chunk_test.Coord1,chunk_test.data['density/number'])
    # plt.show()
    # chunk_test.centre()
    # plt.plot(chunk_test.Coord1,chunk_test.data['density/number'])

    # plt.show()
    chunk_test.centre()
    # test getter
    print(chunk_test.Coord1)
    #test setter
    chunk_test.Coord1 = chunk_test.Coord1*2

    print(chunk_test.Coord1)

    print(chunk_test.density_mass)
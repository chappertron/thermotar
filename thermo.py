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

class Thermo():

    def __init__(self,thermo_df,file2 = None,CLEANUP = True, **kwargs):

        ''' thermo_file - string of log file location '''
        self.data : pd.DataFrame = thermo_df

        # clean up dataframe

        #apply strip_pref function to remove 'c_/f_/v_' prefixes to all columns
        if CLEANUP:
            self.data.rename(columns = lmp_utils.strip_pref, inplace = True)

            #todo merge columns into vectors

        
        for col in self.data.columns:
            setattr(self, col ,getattr(self.data, col))
        
    
    @classmethod
    def create_thermos(cls,logfile, join = True):
        '''utilises parse_thermo to create thermo objects'''
        
        # make load thermos as  IO objects
        strings_ios = Thermo.parse_thermo(logfile, f = StringIO )
        # load io strings as dataframes and return as thermo object
        if not join:
        
            return [Thermo(pd.read_csv(csv, sep='\s+') ) for csv in strings_ios]

        else:
            joined_df = pd.concat([pd.read_csv(csv, sep='\s+') for csv in strings_ios]).reset_index()

            return Thermo(joined_df)

    @classmethod
    def parse_thermo(cls,logfile, f = None ):
        '''
        Parses the given LAMMPS log file and outputs a list of strings that contain each thermo time series.
        # Change so outputs a list of thermo objects
        Optional argument f is applied to list of strings before returning, for code reusability
        
        '''
        #todo perhaps change to output thermo objects???
        #todo add automatic skipping of lines with the wrong number of rows
        #todo if no 'Per MPI rank' found treat as if the file is a tab seperated file

        thermo_datas = []
    
        with open(logfile,'r') as stream:
            current_thermo = ""
            thermo_start = False

            # Parse file to find thermo data, returns as list of strings
            for line in stream:
                if line.startswith(r'Per MPI rank'):
                    current_thermo = ""
                    thermo_start = True
                elif line.startswith(r'Loop time of'):
                    thermo_datas.append(current_thermo)
                    thermo_start = False
                elif thermo_start:
                    current_thermo += line

        if thermo_start:
            # if the thermo series is incomplete appends it anyway
            thermo_datas.append(current_thermo) 

        # if len(thermo_datas) ==0:
        #     try:
        #         #load as file

        if f:
            return [ f(i) for i in thermo_datas] #applies function to string, default is to do nothing
        else:
            return thermo_datas
            
    @staticmethod
    def split_thermo (logfile, path = './split_thermos/',file_name_format='thermo_{}.csv',**kwargs):
        #todo make class method???
        thermo_lists = Thermo.parse_thermo(logfile)
    
        try:
            os.mkdir(path)
        except FileExistsError:
            pass


        for i,thermo in enumerate(thermo_lists):
            out_file = open(path+file_name_format.format(i),'w')
            out_file.write(thermo)
            out_file.close()

        return thermo_lists


    def plot_property(self,therm_property,x_property = None, **kwargs):
        
        #todo allow plotting many properties at once   

        the_data = self.data

        if therm_property not in the_data.keys():
            raise KeyError(f'Property {therm_property} not found')

        if x_property == None:
            if 'Step' in self.data.keys():
                x_property = 'Step'
            elif 'Time' in self.data.keys():
                x_property = 'Time'
            else: 
                x_property = None

        #print('got this far!!!')


        return self.data.plot(x = x_property,y = therm_property,ls=' ',marker='x',**kwargs)

    def reverse_cum_average(self,property):
        prop = self.data[property]

        cum_ave=np.array([np.mean(prop[i:]) for i, point in enumerate(prop)])

        return pd.Series(cum_ave)

    def compare_dist(self,property,bins=100,**kwargs):
        self.data[property].plot.density(**kwargs)
        self.data[property].plot.hist(**kwargs,density=True,bins=bins)
        
        return plt.axis

if __name__ == "__main__":
    print(Thermo.split_thermo('my.log'))
    test_thermo = Thermo('split_thermos/thermo_3.csv')
    print(test_thermo.data)
    print(test_thermo.plot_property('Temp'))

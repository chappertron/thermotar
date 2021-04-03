''' Defines a thermo class
    Thermo Data is extracted from log files
'''

from thermotar.utils import parse_logs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from io import StringIO
# from .utils import lmp_utils
from .utils import lmp_utils
from .utils import df_utils
class Thermo():

    def __init__(self,thermo_df,file2 = None,CLEANUP = True,properties=None, **kwargs):

        ''' thermo_df - data frame containig thermodynamic info time serieses
            CLEANUP - Option to remove c_ etc. prefixes from column names and stuff
            
            properties - dict of properties paresed from the log file. Used in create thermos or the get_props class method

             '''
        self.data : pd.DataFrame = thermo_df

        # clean up dataframe

        
        if CLEANUP:
            #apply strip_pref function to remove 'c_/f_/v_' prefixes to all columns
            self.data.rename(columns = lmp_utils.strip_pref, inplace = True)
            # replace '/' and '[]' as well as other python unfriendly ccharacters,so these columns can be accessed with attributes
            self.data.rename(columns=lmp_utils.drop_python_bad, inplace= True)

        self.properties_dict = properties

        if self.properties_dict:
            # set up properties

            self.time_step = self.properties_dict['time_step']
            self.box = self.properties_dict['box']
            # called box_Lx rather than Lx incase it is reported via thermo output
            self.box_Lx = self.box[3] - self.box[0]
            self.box_Ly = self.box[4] - self.box[1]
            self.box_Lz = self.box[5] - self.box[2]
            self.lattice_initial = self.properties_dict['lattice_initial']


        
        # for col in self.data.columns:
        #     setattr(self, col ,getattr(self.data, col))
        # sets setters and getters for each column of the df as attributes of the CLASS. Has to be class, not the object itself
        df_utils.raise_columns(self)


    def heat_flux(self,thermostat_C='thermostatC',thermostat_H='thermostatH', area = None, style = 'linear',axis='z',C_H_ratio = 1.0,method='linear_fit',direction = 1, real_2_si = True):
        '''
            thermostat_C  - str:
                Column name of the cold thermostat energy removal
            thermostat_H - str:
                Column name of the hot thermostat compute
            Area - None, float,array:
                if None, work out cross sectional area from properties, if a float, assumes constant area along the axis,
                if an array, take values. If style is radial, and a float, this is taken to be the radius of the device 
                Default - None
            style - str:
                Can be linear or radial atm - the geometry of the syste,
                default: linear
            axis - str
                    Name of axis along which heat flux is applied
                    default 'z'

            C_H_ratio - float:
                ratio between the volume of the cold to the volume of the hot thermostats, to normalise heat flow.

            direction - hot to cold = 1,  cold to hot = -1 - matches the sign of the thermal gradient

        '''
        # for spheriical, area needs to be a radius or an array of points for the area as a function of r

        if area is None:
            # find the area if it has been located in the thermo file metadata
        
            if axis == 'x':
                area = self.box_Ly*self.box_Lz
            elif axis == 'y':
                area = self.box_Lx*self.box_Lz
            elif axis == 'z':
                area = self.box_Lx*self.box_Ly
            else:
                raise ValueError('axis must be x,y, or z')
            try:
                time = self.time
            except:
                time = self.time_step*self.Step
            


        if method == 'linear_fit':
            fit_H = np.polyfit(time, direction*self.data[thermostat_H],1) # so heat flows from hot to cold
            fit_C = np.polyfit(time,-1*direction*self.data[thermostat_C],1) # -1 * thermostat Cso heat flows from hot to cold
            
            # average the hot and cold thermostats # second divide by 2 is accounting for the fact there are 2 fluxes in the box
            e_flow = (fit_H[0]+fit_C[0])/2/2 

        if real_2_si:
            kcal_per_mol = 4.184e3 / 6.02214076e23 # J # 1 kcal
            # factor of 1e15 below converts to per s rather than per fs
            # multiplication by 1e20 makes per m2 rather than per ang strom sq
            return e_flow * kcal_per_mol * 1e15 /area /(1e-20)

            
        return e_flow / area
    
    @classmethod
    def create_thermos(cls,logfile, join = True, get_properties = True, last = True):
        '''
        utilises parse_thermo to create thermo objects
        
        join: boolean - 
            Decide whether to concatenate the thermo output of different run commands into one df or not
            If False a list of thermo objects is returned
            default: True
        last: Just get the last set of data, usually production steps.
        
        
        '''
        
        # make load thermos as  IO objects
        strings_ios = Thermo.parse_thermo(logfile, f = StringIO )
        # load io strings as dataframes and return as thermo object
        
        if get_properties:
            properties = Thermo.get_properties(logfile)
        else:
            properties = None
        
        if last:
            return Thermo(pd.read_csv(strings_ios[-1], sep='\s+'),properties=properties) 
        if not join:
        
            return [Thermo(pd.read_csv(csv, sep='\s+'),properties=properties ) for csv in strings_ios]

        else:
            joined_df = pd.concat([pd.read_csv(csv, sep='\s+') for csv in strings_ios]).reset_index()

            return Thermo(joined_df,properties=properties)

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


    @staticmethod
    def get_properties(logfile):
        ''' From a log file get properties defined in parse_logs.py in lmp_properties_re'''

        properties_dict = parse_logs.get_lmp_properties(logfile)

        return properties_dict


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

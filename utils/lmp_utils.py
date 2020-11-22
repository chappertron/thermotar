# utilities for cleaning up dataframes from lammps output

import pandas as pd
import re

# For renaming columns
# 
# 
# Striping prefixes from column names

def strip_pref(col_name):
    # applied when read in
    # re for searching whether the df
    # for matching 'c_', 'f_' or 'v_' at the beginning of a string
    prefix_re = re.compile(r'^[cfv]_')

    col_stripped = prefix_re.sub('',col_name)

    return col_stripped





def raise_attribute(thermolike):
    '''input obj has to have a .data attribute'''
    for col in thermolike.data.columns:
        setattr(thermolike, col ,getattr(thermolike.data, col))
    

def gmx_drop_units(header):
    '''
        GROMACS .xvgs have the header format : property_name \s+ (Units) 
        This function deletes the bracketed units term and then removes leading and trailing whitespace.
    '''
    pass
    


    



# vectorise take columns x[1] x[2] x[3]  and combine into one column of python objects 
# (lists or ndarray???)



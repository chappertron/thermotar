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

def drop_python_bad(col_name):
    '''
        Remove characters from the column names (input string) that don't work in python varialbe names
    '''
    
    
    # applied when read in
    # re for searching whether the df
    # for matching 'c_', 'f_' or 'v_' at the beginning of a string
    
    left_bracket_re=re.compile(r'[<[{(]') # workout how to replace with _contents of this... # no backslashes needed because with in the group
    
    right_bracket_re=re.compile(r'[>})\]]') # workout how to replace with _contents of this... # no backslashes needed because with in the group

    non_python_re = re.compile(r'\W') #matches any non word character

    col_name = left_bracket_re.sub('_',col_name)
    col_name = right_bracket_re.sub('',col_name) # replace with empty string so there is only an underscore before the thing, not before and after.

    col_stripped = non_python_re.sub('_',col_name) # replace any matched character with an underscore

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



import re
import numpy as np
import pandas as pd
from . import lmp_utils


''' TODO fix match units(xvg), raising errors if there are 
less units than unitless coulmns (such as in the case of energy.xvg with 
multiple energies)'''


def parse_lammps():
    pass



# making more general (picking out headers and legends seperately)
def _parse_xvg(file, return_units=True, match_units = False):
    ''' 
        Parse GROMACS .xvg output files and generate a numpy array, a dictionary of x/y labels and units and legends and units
        
        file: str with path location or fileIO

    '''

    xy_re = r'@\s+([xy]axis)\s+(label) "(?P<header>[\w\s\\]+)(|\((?P<units>.+)\))"'
    xy_units_list_re = r'@\s+[xy]axis\s+label\s+"(\((?P<units>[^()]+)\)[,\s]*)+"'  # looks for (with units in and one or more)
    units_re = r'(\((?P<units>[^()]+)\)[,\s]*)'
    legend_re = r'@\s+(s\d+)\s+(legend) "(?P<header>[\w\s\.\\]+)(|\((?P<units>.+)\))"'

    data = np.loadtxt(file,comments = ['#','@','&'])
    labels = dict()
    legends = dict()
    units_list = []

    with open(file,'r') as stream:
        for line in stream:
            #look for title lines
            if not (line.startswith('@') or line.startswith('#')):
                # if gone on to the main file, stop iterating through
                break
            if match := re.match(xy_re,line):
                # find the header rows. If match is found, append header and units/
                try:
                    labels[match.group('header').strip()]=match.group('units').strip()
                except AttributeError:
                    labels[match.group('header').strip()]="Unitless"
            if match := re.match(legend_re,line):
                # find the header rows. If match is found, append header and units/
                try:
                    legends[match.group('header').strip()]=match.group('units').strip()
                except AttributeError:
                    legends[match.group('header').strip()]="Unitless"
            if (match := re.match(xy_units_list_re,line)) and match_units:
                # if the y-axis is of this form, assuming it gives the units of the legend labels, like in the outputs of gmx energy
                units_list = [m.group('units') for m in re.finditer(units_re,line)]

                # combine units with undefined units
                print('here')
               
    if match_units: # match up the units with the associated key. assumes all units for all legends are given in the y axis
        for i, key in enumerate(legends.keys()):
            if legends[key]=="Unitless":
                try:
                    legends[key] = units_list[i]  
                except IndexError:
                    # perhaps implement warning here that match units was specified, despite no units being found
                    # currently just setsunits to last unit in list. Won't be correct usually
                    legends[key] = units_list[-1]
   
    df = pd.DataFrame(data)

    return df, labels, legends


def parse_xvg(file,style='auto',units = False,match_units=False,**kwargs):
    
    df, labels, legends = _parse_xvg(file,match_units=match_units,**kwargs)


    if style == 'auto':
        # combine 
        headers = {**labels, **legends}
        try:
            df.columns = headers.keys()
            unit_tables = headers
        except ValueError:
            # TODO implement alternative to adding all legends and labels for more than cols!
            style = 'rdf'
    if style == 'dipoles':
        headers = {**labels, **legends}
        df.columns = headers.keys()
        unit_tables = headers

    if style == 'rdf':
        # rdf files have x and y labels and legends: solution
        # combine y and legends
        if len(labels) == 2:
            ## TODO fix this very unelegant solution of casting to dicts
            xlabel = dict([list(labels.items())[0]])
            ylabel = dict([list(labels.items())[1]])
        yname = list(ylabel)[0]
        legends_joined = {yname+'_'+key:val for key,val in legends.items()}

        headers = {**xlabel, **legends_joined}
        df.columns = headers.keys()
        unit_tables = headers

    if units:
        return df, unit_tables
    return df

if __name__ == "__main__":

    ## testing on example xvg files

    xvg_test = _parse_xvg('test_files/energy.xvg')

    print(xvg_test[0].head())
    print(xvg_test[1],'\n',xvg_test[2])
    [print(i) for i,_ in enumerate(xvg_test[2].keys())]

    xvg_test_2 = parse_xvg('test_files/energy.xvg')

    print(xvg_test_2.head())

    # df = xvg_test[0]
    # column_names = {**xvg_test[1], **xvg_test[2]}.keys()
    # df.columns =list(column_names)
    # print(df.head())




'''OLD VERSIONS OF FUNCTIONS'''

# def parse_xvg(file, return_units=True):
#     ''' 
#         Parse GROMACS .xvg output files and generate a dataframe
        
#         file: str with path location or fileIO

#     '''
#     #header_re = r'@\s+[xy]axis\s+label\s+"(?P<header>)\s+ \((?P<units>)\)"'

#     #header_re = r'^@\s+(s\d+|[xy]axis)\s+(legend|label)\s+"(?P<header>.+) (|\((?P<units>.+)\))"'
#     header_re = r'@\s+(s\d+|[xy]axis)\s+(legend|label) "(?P<header>[\w\s\\]+)(|\((?P<units>.+)\))"'

#     # '@ s0 legend "epsilon"'
    
#     # (?P<header> . )
#     header_legend_re = r'@ s0 legend "(?P<header>[\w\s]+)"'

#     # '@    xaxis  label "Time (ps)"'

#     # 


#     data = np.loadtxt(file,comments = ['#','@','&'])
#     headers = []
#     units = []

#     with open(file,'r') as stream:
#         for line in stream:
#             #look for title lines
#             if not (line.startswith('@') or line.startswith('#')):
#                 # if gone on to the main file, stop iterating through
#                 break
#             if match := re.match(header_re,line):
#                 # find the header rows. If match is found, append header and units/
#                 headers.append(match.group('header').strip())
#                 try:
#                     units.append(match.group('units').strip())
#                 except AttributeError:
#                     units.append('Unitless')
                    
#     if len(headers)==0:
#         print('header search failed')
#         df = pd.DataFrame(data)
#     else:
#         #print(data.shape,len(headers))
#         df = pd.DataFrame(data,columns=headers)

#     return df
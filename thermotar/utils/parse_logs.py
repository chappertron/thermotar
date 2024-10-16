import re
import numpy as np
import pandas as pd

# from t import lmp_utils
from collections import defaultdict

""" TODO fix match units(xvg), raising errors if there are 
less units than unitless columns (such as in the case of energy.xvg with 
multiple energies)"""


def parse_lammps():
    pass


# list of properties for to find in the lammps log files. Dict that maps to target re.

lmp_property_re = {
    "box": r"[\s]*[\w]+ box = \((?P<xlo>-?[\d.]+(?:e-?\d+)?)\s+(?P<ylo>-?[\d.]+(?:e-?\d+)?)\s+(?P<zlo>-?[\d.]+(?:e-?\d+)?)\) to \((?P<xhi>-?[\d.]+(?:e-?\d+)?)\s+(?P<yhi>-?[\d.]+(?:e-?\d+)?)\s+(?P<zhi>-?[\d.]+(?:e-?\d+)?)\)",
    "lattice_initial": r"Lattice spacing in x,y,z\s+=\s+(?P<group>[\d.]+)\s+(?P<group1>[\d.]+)\s+(?P<group2>[\d.]+)",
    "time_step": r"\s*Time step\s+:\s+(?P<dt>[\d.]+)",
}


def get_lmp_properties(file, verbose=False):
    """Parse a log file and get various initial/final/global properties of the system"""
    # test_re = lmp_property_re['lattice_initial']
    # test_re = lmp_property_re['box']
    properties = defaultdict(None)

    match = None
    for key, re_str in lmp_property_re.items():
        if verbose:
            print(key, re_str)

        test_re: re.Pattern = re.compile(re_str)

        with open(file) as stream:
            for i, line in enumerate(stream):
                # print(f'starting to look at {i}')
                match = test_re.match(line)
                # print(match)
                if match is not None:
                    if verbose:
                        print(f"made a match on line {i}")
                    if verbose:
                        print(np.array(match.groups()).astype(float))
                    properties[key] = np.array(match.groups()).astype(float)
                    break  # if found move on to the next property
    return properties


# making more general (picking out headers and legends separately)
def _parse_xvg(file, return_units=True, match_units=False):
    """
    Parse GROMACS .xvg output files and generate a numpy array, a dictionary of x/y labels and units and legends and units

    file: str with path location or fileIO

    """

    xy_re = r'@\s+([xy]axis)\s+(label) "(?P<header>[\w\s\\]+)(|\((?P<units>.+)\))"'
    xy_units_list_re = r'@\s+[xy]axis\s+label\s+"(\((?P<units>[^()]+)\)[,\s]*)+"'  # looks for (with units in and one or more)
    units_re = r"(\((?P<units>[^()]+)\)[,\s]*)"
    legend_re = r'@\s+(s\d+)\s+(legend) "(?P<header>[\w\s\.\\]+)(|\((?P<units>.+)\))"'

    data = np.loadtxt(file, comments=["#", "@", "&"])
    labels = dict()
    legends = dict()
    units_list = []

    with open(file, "r") as stream:
        for line in stream:
            # look for title lines
            if not (line.startswith("@") or line.startswith("#")):
                # if gone on to the main file, stop iterating through
                break
            if match := re.match(xy_re, line):
                # find the header rows. If match is found, append header and units/
                try:
                    labels[match.group("header").strip()] = match.group("units").strip()
                except AttributeError:
                    labels[match.group("header").strip()] = "Unitless"
            if match := re.match(legend_re, line):
                # find the header rows. If match is found, append header and units/
                try:
                    legends[match.group("header").strip()] = match.group(
                        "units"
                    ).strip()
                except AttributeError:
                    legends[match.group("header").strip()] = "Unitless"
            if (match := re.match(xy_units_list_re, line)) and match_units:
                # if the y-axis is of this form, assuming it gives the units of the legend labels, like in the outputs of gmx energy
                units_list = [m.group("units") for m in re.finditer(units_re, line)]

                # combine units with undefined units
                print("here")

    if match_units:  # match up the units with the associated key. assumes all units for all legends are given in the y axis
        for i, key in enumerate(legends.keys()):
            if legends[key] == "Unitless":
                try:
                    legends[key] = units_list[i]
                except IndexError:
                    # perhaps implement warning here that match units was specified, despite no units being found
                    # currently just setsunits to last unit in list. Won't be correct usually
                    legends[key] = units_list[-1]

    df = pd.DataFrame(data)

    return df, labels, legends


def parse_xvg(file, style="auto", units=False, match_units=False, **kwargs):
    """
    Parse GROMACS .xvg output files and generate a headered dataframe and optionally also a table of units for the columns

    file: str with path location or fileIO

    """

    df, labels, legends = _parse_xvg(file, match_units=match_units, **kwargs)

    if style == "auto":
        # combine
        headers = {**labels, **legends}
        try:
            df.columns = headers.keys()
            unit_tables = headers
        except ValueError:
            # TODO implement alternative to adding all legends and labels for more than cols!
            style = "rdf"
    if style == "dipoles":
        headers = {**labels, **legends}
        df.columns = headers.keys()
        unit_tables = headers

    if style == "rdf":
        # rdf files have x and y labels and legends: solution
        # combine y and legends
        # print(labels)
        # print(legends)
        if (
            len(labels) == 2
        ):  # double check both an x and y are found; kinda required for this style
            ## TODO fix this very unelegant solution of casting to dicts
            xlabel = dict([list(labels.items())[0]])
            ylabel = dict([list(labels.items())[1]])
        yname = list(ylabel)[0]

        # pickout the name of each legend and append to y
        # TODO, make units of y not units of the legend!!!
        # nb this works for the rdf style but not energy
        legends_joined = {yname + "_" + key: val for key, val in legends.items()}

        headers = {**xlabel, **legends_joined}

        print("headers: ", headers)

        unit_tables = headers

    if units:
        return df, unit_tables

    df.columns = headers
    return df


if __name__ == "__main__":
    ## testing on example xvg files

    xvg_test = _parse_xvg("test_files/energy.xvg")

    print(xvg_test[0].head())
    print(xvg_test[1], "\n", xvg_test[2])
    [print(i) for i, _ in enumerate(xvg_test[2].keys())]

    xvg_test_2 = parse_xvg("test_files/energy.xvg")

    print(xvg_test_2.head())

    # df = xvg_test[0]
    # column_names = {**xvg_test[1], **xvg_test[2]}.keys()
    # df.columns =list(column_names)
    # print(df.head())


"""OLD VERSIONS OF FUNCTIONS"""

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

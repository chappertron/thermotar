import aiofiles
from .thermo import Thermo 
import warnings
from typing import Union, List
from io import StringIO
import pandas as pd

async def async_parse_thermo(
    logfile,
):
    """
    Asynchronously parses the given LAMMPS log file and outputs a list of strings 
    that contain each thermo time series.
    """
    thermo_datas = []
    async with aiofiles.open(logfile, "r") as stream:
        current_thermo = ""
        thermo_start = False
        warnings_found = False

        # Parse file to find thermo data, returns as list of strings
        async for line in stream:
            if line.startswith(
                r"Per MPI rank"
            ):  #### todo use regular expression instead???
                current_thermo = ""
                thermo_start = True
            elif (
                line.startswith(r"Loop time of")
                or line.startswith(r"ERROR")
                or line.startswith("colvars: Saving collective")
            ):  ### this used to create a bug when the thing errors out
                thermo_datas.append(current_thermo)
                thermo_start = False
            elif line.startswith("WARNING"):
                # If the line is a warning, skip to the next line
                warnings_found = True
                continue
            elif thermo_start:
                current_thermo += line

        if thermo_start:
            # if the thermo series is incomplete appends it anyway
            thermo_datas.append(current_thermo)

        # TODO Gather warnings and emit them
        if warnings_found:
            warnings.warn("Warnings found when reading File")

    # if len(thermo_datas) ==0:
    #     try:
    #         #load as file

    return thermo_datas


async def async_create_thermos(
    logfile, join=True, get_properties=False, last=True
) -> Union[List[Thermo], Thermo]:
    """
    Asynchronously parses the given LAMMPS log file and outputs a list of
    thermo objects.
    Uses aiofiles to read the file asynchronously.
    """
    strings_ios = await async_parse_thermo(logfile)
    if get_properties:
        raise NotImplementedError("Properties not implemented yet for async version")
    properties = None

    if last:
        return Thermo(
            pd.read_csv(StringIO(strings_ios[-1]), sep=r"\s+"), properties=properties
        )
    if not join:
        return [
            Thermo(pd.read_csv(StringIO(csv), sep=r"\s+"), properties=properties)
            for csv in strings_ios
        ]

    else:
        joined_df = pd.concat(
            [pd.read_csv(StringIO(csv), sep=r"\s+") for csv in strings_ios]
        ).reset_index()

        return Thermo(joined_df, properties=properties)
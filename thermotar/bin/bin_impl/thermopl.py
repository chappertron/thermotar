#! /usr/bin/env python


from pathlib import Path
from typing import Dict, Tuple
from enum import Enum, auto
from typing_extensions import Self
import typing

from numpy.linalg import LinAlgError
import pandas as pd


def fit(x: pd.Series, y: pd.Series) -> Tuple[Tuple[float, float], pd.Series]:
    """
    Fit the data and return the fittings + the values of the fitted line.
    """
    import numpy as np

    try:
        params = np.polyfit(x, y, 1)
    except LinAlgError:
        return (np.nan, np.nan), pd.Series(np.nan, index=x.index)

    vals = np.polyval(params, x)

    return (params[0], params[1]), pd.Series(vals)


class Units(Enum):
    """
    Which set of LAMMPS units are we using?

    Current plan is to use to index a Dict. Maybe a rust style enum may be better?
    """

    REAL = auto()
    LJ = auto()

    @classmethod
    def from_str(cls, s: str) -> Self:
        """
        Get units enum from a string
        """

        s = s.upper()

        return Units[s]


# units  =

# TODO use this state class, or not.
# @dataclass()
# class State:
#     properties_in_all: Set[str]
#     properties_in_some: Set[str]
#     dfs: pd.DataFrame

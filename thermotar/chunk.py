"""Class for Reading output from LAMMPS chunk/ave command."""

from os import PathLike
from typing import Optional, List, Union
import numpy as np
import pandas as pd

# from .utils import lmp_utils
from .utils import lmp_utils
from .utils import parse_chunks
from .utils import df_utils
from .utils.df_utils import raise_col


class Chunk:
    """Class for Reading output from LAMMPS chunk/ave command."""

    def __init__(
        self,
        thermo_df: pd.DataFrame,
        CLEANUP=True,
        coord_cols=["Coord1", "Coord2", "Coord3", "coord", "Box"],
        centred=False,
        centered=None,
        **kwargs,
    ):
        """
        Construct a `Chunk`  from a pandas Dataframe.

        Parameters
        ----------
        thermo_df :
            Data frame to read values from.
        CLEANUP :
            If true, the headers of the DataFrame are tided up to become valid python
            identifiers and strips the prefixes from compute/fix and variable columns.
        centred :
            Whether the coordinates of the system are already centred. This option
            will be deprecated; the centring calculation is cheap.

        """
        self.data: pd.DataFrame = thermo_df

        # clean up dataframe

        # apply strip_pref function to remove 'c_/f_/v_' prefixes to all columns
        if CLEANUP:
            # TODO: the prefix stripping should not be done if there ends up being
            # ambiguity between the fields; for example if c_foo and f_foo are both
            # defined
            self.data.rename(columns=lmp_utils.strip_pref, inplace=True)
            self.data.rename(columns=lmp_utils.drop_python_bad, inplace=True)

        # set the columns as attributes
        for col in self.data.columns:
            # has to be set to a method of the class
            setattr(
                self.__class__, col, raise_col(self, col)
            )  # set attribute to this property object??
        # column names for the coordinates, up to 3
        # only those in the df are included, by finding intersect of sets.
        self.coord_cols = list(set(self.data.columns.to_list()) & set(coord_cols))
        if centered is not None:
            centred = centered

        self.centred = centred  # Initialise assuming asymmetrical - to do implement a method to check this!!!!

    # Property Definitions

    @property
    def centered(self) -> bool:
        """Return whether the coordinates have already been centred."""
        return self.centred

    @classmethod
    def create_chunk(
        cls, fname: Union[str, PathLike], style: str = "lmp", last: bool = True
    ):
        """
        Load LAMMPS or numpy savetxt as a df and then create a Chunk instance.

        Parameters
        ----------
        fname:
            File to load.
        style:
            What is the format of the file.
            Supported values are "lmp" and "np", for lammps chunkfiles and numpy
            savetxt output, respectively.
        last:
            Only true is supported. Whether to read the last frame of the chunk file
            or all of them. Use `MultiChunk` if all frames are needed.
        """
        # TODO: Implement .xvg styles
        # TODO: Implement auto choosing the style, either with file extensions
        # or some form of try and fail

        # if style == 'auto':
        #     try:
        #         with open(fname,'r') as stream:
        #             for i,line in enumerate(stream):
        #                 if i==3: int(line.split()[1]) # try to cast into an integer, if this fails, likely not lmp style
        #         style = 'lmp'
        #     except ValueError:
        #         # set style = 'np'
        #         style = 'np'
        if style == "lmp":
            if last:
                df = parse_chunks.lmp_last_chunk(fname, nchunks="auto", header_rows=3)
            else:
                raise NotImplementedError(
                    "Only the last chunk is supported at the moment."
                )
        elif style == "np":
            # read the data frame from a space separated file outputted by np.savetxt.
            df = parse_chunks.parse_numpy_file(fname, header_row=0, comment="#")
        else:
            raise NotImplementedError(
                "Only LAMMPS chunk files and numpy save_txt files are supported so far."
            )

        return cls(df)

    def raise_columns(self):
        """Raise columns from the df to be attributes."""
        # I have no clue how pandas does this automatically...
        # Maybe I need to make it so my objects can be indexed
        # and in doing that for assignment, the attributes can be raised up
        # TODO : Something with above??
        df_utils.raise_columns(self)

    def prop_grad(self, prop: str, coord: str, **kwargs):
        """Calculate the gradient of `prop` with respect to `coord`.

        Parameters
        ----------
        prop:
            Which property the gradient of is calculated
        coord:
            Which property is used for the coordinate.
        kwargs:
            Keyword arguments to pass to `np.gradient`
        """
        df = self.data

        df[prop + "_grad"] = np.gradient(df[prop], df[coord], **kwargs)

        # updates the columns
        df_utils.raise_columns(self)

    def nearest(self, property: str, value: float, coord=0):
        """Return the index for which property is closest to `value`.

        To do, return the actual row of properties??????

        This way if nan - make a nan list

        """
        # TODO: Find out where this method is used and understand what on earth it does
        coord = self.data[
            self.coord_cols[coord]
        ]  # select coordinate column for which to find nearsest in each side

        prop = self.data[property]
        # if each half, find the index in each half of the simulation box

        left = prop.loc[coord < 0]
        right = prop.loc[coord > 0]

        right_nearest_index = right.sub(value).abs().idxmin()

        if len(left.index) != 0:
            left_nearest_index = left.sub(value).abs().idxmin()
            indexes = [left_nearest_index, right_nearest_index]
        else:
            indexes = [right_nearest_index]

        try:
            return self.data.loc[indexes].copy()
        except KeyError:  # just create an empty row
            row_copy = self.data.loc[
                (0,), :
            ].copy()  # janky indexing makes it a df rather than a series
            row_copy.loc[:] = np.nan
            return row_copy

    def centre(
        self, coord: Union[int, str, List[str]] = "all", moment: Optional[int] = None
    ):
        """Shift the origin of the simulation box to zero.

        Parameters
        ----------
        coord:
            Index of coordinate column to centre , indexes self.coord_cols.
            Default is 'all'.

        moment:
            If Not None, centres the system to this column name,
            weighted by this column name raised to the power of moment.

        """
        if coord == "all":
            # calculate the union of the list of coord_cols and the df columns
            coords = self.coord_cols

        elif isinstance(coord, int):
            coords = [self.coord_cols[coord]]
        else:
            # if neither a number or 'all', assumes a column name
            coords = [coord]
        # iterate over selected coordinates and perform the centring operation
        for coord_col in coords:
            if moment:
                self.data[coord_col] -= self.moment(
                    coord_col, moment
                )  # set origin to be first moment, weighted by moment parameter
            else:
                # print(coord_col)
                self.data[coord_col] = self.centre_series(self.data[coord_col])

        self.centred = True
        # Return the object at the end for method chaining
        return self

    def center(self, coord="all"):
        """An alias of centre for yanks."""
        return self.centre(coord=coord)

    @staticmethod
    def centre_series(series: pd.Series):
        """Subtract the average of a series from the series."""
        centre = (series.max() + series.min()) / 2
        centred = series - centre
        return centred

    def parity(self, prop, coord=0):
        """
        Multiplies a property by the sign of the coordinate.

        Should only be applied to properties that are pseudo scalars,  i.e. change
        sign under coordinate inversion, so that upon folding and averaging
        properties are correct.
        """
        # centre first
        if isinstance(coord, int):
            coord = self.coord_cols[coord]
        if not self.centred:
            self.centre()

        self.data[prop + "_original"] = self.data[prop]
        self.data[prop] *= np.sign(
            self.data[coord]
        )  # multiply by the sign of the coordinate column
        self.raise_columns()

    def moment(
        self,
        coord,
        weighting,
        order=1,
    ):
        """Calculate the specified moment of the coordinate, weighted by a named property"""
        coords = self.data[self.choose_coordinate(coord)].T

        integrand = self.data[weighting] * coords**order
        normaliser = np.trapz(self.data[weighting], coords)

        return np.trapz(integrand, coords) / normaliser

    def choose_coordinate(self, coord: Union[int, str]):
        """Find the provided coordinate column(s).

        If an integer, indexes the self.coord_cols field,
        if string 'all', returns self.coord_cols
        if any other, returns
        """
        if coord == "all":
            # calculate the union of the list of coord_cols and the df columns
            coords = self.coord_cols

        elif isinstance(coord, int):
            coords = [self.coord_cols[coord]]
        else:
            # if neither a number or 'all', assumes a column name
            coords = [coord]

        return coords

    def fold(self, crease=0.0, coord=None, coord_i: int = 0, sort=False, inplace=True):
        """
        Fold the profile about coord = crease.

        WARNING: if properties have been calculated prior to folding, they may no
        longer be correct.

        For example, electric fields calculated by integrating charge profiles,
        will have a different sign in each part of the box.

        To deal with this they should be inverted based on the sign of the coordiante.

        Parameters
        ----------
        crease: -
                Position along folded coordinate to fold about

        coord_i : int -
                The index of the self.coord_cols list. Default is 0, the first coord
        """
        if coord is None:
            coord_name = self.coord_cols[coord_i]
        else:
            coord_name = coord

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
            self.data[coord_name] -= (
                crease  # set origin to the value to be creased about
            )
            self.data[coord_name] = np.absolute(
                self.data[coord_name]
            )  # get absolute value
            self.data[coord_name] += (
                crease  # add the crease value back so still starts at this value!
            )

        # TODO Implement fully, even when not centred!

        # TODO auto sort ?

        if sort:
            # sorts the data by this column name
            self.data.sort_values(
                by=coord_name, inplace=True, ignore_index=True
            )  # labels 0,1,2.... Needed for future integration/differentaion operations in numpy

        pass

    def rebin(
        self,
        coord,
        bw=0.25,
        bins=None,
        nbins=None,
        mode="average",
        inplace=False,
        new_coord_loc="mid",
        weights=None,
    ):
        """
        Rebin the data based on coordinates for a given new bin width.

        Default is to perform averages over these bins.
        Could also add weightings for these averages

        Parameters
        ----------
        coord:
            Column name of the coordinate to create the bins from

        nbins:
            None, int, or array of ints. Number of bins for each binning dimension.
            Currently only supported for 2D bins.

        new_coord_loc:
            "mid" (default), "left" or "right" position of new coordinate,
            when set manually rather than from average. Currently only for 2D bins

        inplace : bool
            if True, overwrites the .data method, else just returns the data frame.

        weights:
            Column label for performing a weighted average,
            only used if mode is "average" or "mean"

        """
        # TODO: - implement n_bins argument for 1d bins
        df = self.data

        # Use multiple bins
        if np.iterable(coord) and not isinstance(coord, str):
            number_coords = len(coord)
            if number_coords == 2:
                coord1, coord2 = coord[0], coord[1]
                df_binned = df_utils.rebin_2D(
                    df,
                    coord1,
                    coord2,
                    bw=bw,
                    bins=bins,
                    mode=mode,
                    nbins=nbins,
                    new_coord_loc=new_coord_loc,
                    weight_col=weights,
                )
            elif np.number >= 3:
                raise NotImplementedError(
                    "Binning in more than two dimensions not yet supported"
                )
        else:
            df_binned = df_utils.rebin(
                df, coord, bw=bw, mode=mode, weight_col=weights, bins=bins
            )
        # coord_max = df[coord].max()
        # coord_min = df[coord].min() # finding min and max so it works with gfolded data

        # n_bins = (coord_max-coord_min) // bw #double divide is floor division
        # n_bins = int(n_bins)

        # bins = pd.cut(df[coord],n_bins)

        # df_grouped = df.groupby(bins)

        # if mode == 'average' or mode == 'mean':
        #     df_binned = df_grouped.mean()
        # else:
        #     df_binned = df_grouped.sum()

        # TODO:  Why is df_binned possibly unbound?
        if inplace:
            self.data = df_binned
        else:
            return df_binned

    def fold_and_ave(
        self,
        crease=0.0,
        coord=None,
        coord_i=0,
        sort=True,
        bw: Union[str, float] = "auto",
    ):
        """Fold the profile and average the two halves.

            WARNING: if properties have been calculated prior to folding,
            they may no longer be correct, epsecially properties that invert under coordinate inversion (pseudoscalars)

            For example, electric fields calculated by integrating charge profiles, will have a different sign in each part of the box.

        Parameters
        ----------
        crease:
                Position along folded coordinate to fold about

        coord_i : int
                The index of the self.coord_cols list. Default is 0, the first coord
                if all, will fold all coordinates, but will only crease

        bw:
            Averaging works by rebinning
            If auto, tries to work out the original bin spacing and then groups by this
            If not auto, specify the bin width in distance.

        """
        if coord is None:
            if coord_i == "all":
                coord_names = self.coord_cols
            else:
                coord_names = [self.coord_cols[coord_i]]
        else:
            coord_names = [coord]

        if crease == 0.0 and not self.centred:
            # if the crease is located at coord = 0, but not already centred then - centres
            self.centre(coord=coord_names)

        df = self.data.copy()

        coord1 = df[coord_names[0]]

        if bw == "auto":
            bw = np.abs(
                coord1.iloc[-1] - coord1.iloc[-2]
            )  # if auto work out from the difference of the last 2 points of the coord

        # # only index by the first coord,but flip all?
        # select_a = (df[coord_names[0]] >=  0)
        # select_b = ~select_a

        # df_a = df.loc[select_a].sort_values(by = coord_names[0] ,inplace = True,ignore_index = True)
        # df_b = df.loc[select_b]
        # df_b[coord_names]=df_b[coord_names].abs()
        # df_b.sort_values(by = coord_names[0] ,inplace = True,ignore_index = True)

        # df_ave = pd.concat({'a':df_a,'b':df_b}).mean(level=1)

        # fold the df
        df[coord_names] = (
            np.abs(df[coord_names] - crease) + crease
        )  # assumes already centred for now
        df_ave = df_utils.rebin(df, coord_names[0], bw).sort_values(
            coord_names[0]
        )  # performing the rebinning # then sort by the coord

        return df_ave

    def __getitem__(self, key: str):
        return self.data[key]


if __name__ == "__main__":
    # testing the chunk creator

    chunk_test = Chunk.create_chunk("./test_files/temp.profile")
    # print(chunk_test.coord_cols)
    # plt.plot(chunk_test.Coord1,chunk_test.data['density/number'])
    # plt.show()
    # chunk_test.centre()
    # plt.plot(chunk_test.Coord1,chunk_test.data['density/number'])

    # plt.show()
    chunk_test.centre()
    # test getter
    print(chunk_test.Coord1)
    # test setter
    chunk_test.Coord1 = chunk_test.Coord1 * 2

    print(chunk_test.Coord1)

    print(chunk_test.density_mass)

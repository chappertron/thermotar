from typing import Union, List
import pandas as pd
import io
import numpy as np


def centre_series(series: pd.Series):
    """Subtract the average of a series from the series."""
    centre = (series.max() + series.min()) / 2
    centred = series - centre
    return centred


def df_ave_and_err(
    df: pd.DataFrame, level=0, suffix="_err", err_method="sem"
) -> pd.DataFrame:
    """
    Average a MultiIndexed dataframe over a specified level and get the standard error (default)
    or standard deviation over that level.

    df : pd.DataFrame = input DataFrame to perform operation on

    level : int = Level on which to perform averaging, works out the other levels to use for grouping.
    TODO: Add support for more than one level.

    suffix : str = String to append to columns in the sem/std dataframe

    err_method : str = "sem" (default) or "std" = Whether to calculate standard deviation or standard error of the mean.
    """

    if not isinstance(
        level,
        int,
    ):
        raise ValueError("`level` argument can only be an integer.")

    all_levels = set(range(df.index.nlevels))

    grouping_levels = list(
        all_levels ^ {level}
    )  # ^ operator removes common elements between the two sets, then combines the rest

    grouped_df = df.groupby(level=grouping_levels)

    ave_df = grouped_df.mean()

    def rename_f(x):
        return x + suffix

    if err_method == "sem":
        err_df = grouped_df.sem()
    elif err_method == "std":
        err_df = grouped_df.std()
    else:
        raise ValueError("Only `std` and `sem` are supported for calculating the error")

    return pd.concat([ave_df, err_df.rename(columns=rename_f)], axis=1)


def find_dupes(*dfs: pd.DataFrame):
    """list of all columns in all dfs. Flattened into one list."""
    cols = [col for df in dfs for col in df.columns.tolist()]
    unique_cols = set(cols)

    seen_cols = set()
    duped_cols = list()
    if len(unique_cols) < len(cols):
        for i in cols:
            if i not in seen_cols:
                seen_cols.add(i)
            else:
                duped_cols.append(i)

    return duped_cols


def merge_no_dupes(*dfs):
    """Find duplicated columns and sets said columns to be the index,
    these dfs are then concated and the index reset."""

    duped_cols = find_dupes(*dfs)
    # only attempts to assign index if duped cols found
    if len(duped_cols) > 0:
        [df.set_index(duped_cols, inplace=True) for df in dfs]

    return pd.concat(dfs, axis=1).reset_index()


def comment_header(file, line_no=0, comments="#", delim=" "):
    """

    TODO: Address performance issues!!!! Compare performance to simply calling read.csv but ignoring the # somehow
    TODO: Maybe for i,line in enumerate(readlines) , if line == line_no is better

    Grab a header at specified line that has a specified comment in front and split based on delim. assumes line is in the first few so first chunk is loaded and lines read

    Parameters:
        file : str
                Name of file to extract header from
        line_no : int
                Line number for the header. First line is zero.
                Default : 0

        delim : str
                Separator for the headers
                Default : ' '
        comments : str
                Comment markers
                Default : '#'

    Returns:
        head : list
                List of strings containing header names

    """

    with open(file) as stream:
        first_line = stream.readline()
        if line_no == 0:
            head = first_line
        else:
            buff = io.DEFAULT_BUFFER_SIZE
            line_size = len(
                first_line
            )  # get the size of the first line, to estimate how much of a buffer to give approximately
            stream.seek(0)  # go back to beginning
            # my weird approach
            try:
                head = stream.readlines(
                    buff + (line_size) * (line_no)
                )[
                    line_no
                ]  # read the nth line # reverses at least the default buffer + line size per line
            except IndexError:
                # if out of range, make buffer bigger!
                stream.seek(0)
                head = stream.readlines(buff + (buff + line_size) * (line_no))[line_no]

        # simpler approach?

        # for i,line in enumerate(stream):
        #     if i == line_no:
        #         head = line

    # strip trailing wspace and then split into a list
    head_list = head.strip().split(
        delim
    )  # split the header line at the desired delimiter
    trimmed = [i for i in head_list if i not in comments]

    return trimmed


def new_cols(inpdf, colnames):
    for col in colnames:
        if col not in inpdf.columns:
            inpdf[col] = np.nan  # create an empty column with this name
    return inpdf


def rebin(
    df,
    binning_coord,
    bw=0.25,
    bins=None,
    mode="average",
    weight_col: Union[str, None] = None,
) -> pd.DataFrame:
    """
    Rebin the data based on coordinates for a given new bin width.
    Default is to perform averages over these bins.
    Could also add weightings for these averages

    coord = Column name of the coordinate to create the bins from

    mode  = "average", "mean", "sum" = either average or add the data in each bin.

    inplace : bool
        if True, overwrites the .data method, else just returns the data frame.
    weight_col : str| None = None(default) or string for column used for weighting. Only used for mode ='average'

    """

    """
        Create bins of a coordinate and then group by this coordinate
        - average by this group
    """

    coord = binning_coord

    coord_max = df[coord].max()
    coord_min = df[coord].min()  # finding min and max so it works with gfolded data

    if bins is None:
        # if bin edges not explicitly provided, calculate them from bw and max and min of coord
        n_bins = int((coord_max - coord_min) // bw)  # double divide is floor division
        bins = pd.cut(df[coord], n_bins)
    else:
        bins = pd.cut(df[coord], bins=bins)

    df_grouped = df.groupby(
        bins,
        as_index=False,
        observed=True,  # observed added to avoid depreaction warning
    )  # don't want another column also called coord!!

    if mode == "average" or mode == "mean":
        if weight_col is None:
            df_binned = df_grouped.mean()
        else:
            df_binned = df_grouped.apply(
                lambda x: np.average(x, weights=x[weight_col], axis=0)
            )
    else:
        df_binned = df_grouped.sum()

    return df_binned


def rebin_2D(
    df,
    binning_coord1,
    binning_coord2,
    bw=0.25,
    nbins=None,
    bins=None,
    mode="average",
    weight_col: Union[str, None] = None,
    new_coord_loc="mid",
):
    """
    Rebin the data based on coordinates for a given new bin width.
    Default is to perform averages over these bins.
    Could also add weightings for these averages

    coord = Column name of the coordinate to create the bins from

    mode  = "average", "mean", "sum" = either average or add the data in each bin.

    inplace : bool
        if True, overwrites the .data method, else just returns the data frame.
    weight_col : str| None = None(default) or string for column used for weighting. Only used for mode ='average'
    new_coord_loc = "left","right", "mid" (default)

    """

    """
        Create bins of a coordinate and then group by this coordinate
        - average by this group
    """

    coord1 = binning_coord1
    coord2 = binning_coord2

    coord1_max = df[coord1].max()
    coord1_min = df[coord1].min()  # finding min and max so it works with gfolded data
    coord2_max = df[coord2].max()
    coord2_min = df[coord2].min()

    if np.iterable(bw):
        bw1 = bw[0]
        bw2 = bw[1]
    else:
        bw1 = bw2 = bw

    if not bins and nbins is None:
        # if bin edges not explicitly provided, calculate them from bw and max and min of coord
        n_bins1 = int(
            (coord1_max - coord1_min) // bw1
        )  # double divide is floor division
        bins1 = pd.cut(df[coord1], n_bins1)
        n_bins2 = int(
            (coord2_max - coord2_min) // bw2
        )  # double divide is floor division
        bins2 = pd.cut(df[coord2], n_bins2)
    elif nbins is not None:
        # TODO: Allow for same number of bins in both dimensions without explicitly needing this defined
        n_bins1, n_bins2 = nbins
        bins1 = pd.cut(df[coord1], n_bins1)
        bins2 = pd.cut(df[coord2], n_bins2)
    else:
        # TODO allow for different predefined bins in each direction
        bins1 = bins2 = bins
    # print(n_bins1, n_bins2)
    df_grouped: pd.DataFrameGroupBy = df.groupby(
        by=[bins1, bins2]
    )  # don't want another column also called coord!!

    if new_coord_loc == "right":

        def bin_loc_func(x):
            return x.right

    elif new_coord_loc == "left":

        def bin_loc_func(x):
            return x.left

    elif new_coord_loc == "mid":

        def bin_loc_func(x):
            return x.mid

    else:
        raise ValueError('new_coord_loc must be either "left", "mid" or "right')

    new_coord1 = bins1.apply(bin_loc_func).groupby([bins1, bins2]).max().to_numpy()
    new_coord2 = bins2.apply(bin_loc_func).groupby([bins1, bins2]).max().to_numpy()

    if mode == "average" or mode == "mean":
        if weight_col is None:
            df_binned = df_grouped.mean()
        else:
            df_binned = df_grouped.apply(
                lambda x: pd.Series(
                    np.average(x, weights=x[weight_col], axis=0), index=x.columns
                )
            )
            # df_binned = df_grouped.apply(grouped_by_weighted_ave,weight_col)
    else:
        df_binned = df_grouped.sum()

    df_binned[coord1] = new_coord1
    df_binned[coord2] = new_coord2

    return df_binned.reset_index(drop=True)


def grouped_by_weighted_ave(df_g: pd.DataFrame, weight_col):
    return (df_g * df_g[weight_col]) / df_g[weight_col]


def n_blocks2bw(series: pd.Series, n: int):
    x_min, x_max = series.min(), series.max()
    return (x_max - x_min) / n


def mask_if_equal(
    df: pd.DataFrame,
    target_col: str,
    exclude: Union[List[str], str, None] = None,
    val: float = 0.0,
) -> pd.DataFrame:
    """
    Masks a row if `target_col` is equal to val, excluding columns listed in `exclude`

    df: pd.DataFrame
    target_col : str
    exclude : List of columns or a coulmn to ignore
    """
    if exclude is not None:
        df = df.set_index(exclude, append=True)
    select = df[target_col] == val

    df_masked = df.mask(select)

    if exclude is None:
        return df_masked
    else:
        return df_masked.reset_index(exclude)


if __name__ == "__main__":
    dict1 = {"Time": [1, 2, 3, 4], "data_a": ["a", "b", "c", "d"]}

    dict2 = {"Time": [1, 69, 124, 4, 52, 23], "data_b": ["a", "b", "c", "d", "e", "g"]}

    df1 = pd.DataFrame(dict1)
    df2 = pd.DataFrame(dict2)

    # df_join = pd.concat([df1,df2],axis=1)

    find_dupes(df1, df2)

    df_merge = merge_no_dupes(df1, df2)

    print(df_merge)

    # testing comment_header

    head = comment_header(
        "./test_files/temp.profile", line_no=2, comments="#", delim=" "
    )

    print(head)

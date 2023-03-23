from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import fsolve
from numpy.polynomial import Polynomial
from functools import partial
from dataclasses import dataclass, field
from matplotlib import pyplot as plt


class PairIntersectFinders:
    """
    Namespace collection of methods for find the intersects between DataFrames

        `coarse`:   Finds the interface crudely. Finds the index of the row with the
                    smallest absolute difference between the y_property
        `interp`:   Interpolates the difference in the y property first, then uses fsolve
                    to find a more precise intersection. Uses `coarse` for a first guess.


    """

    @staticmethod
    def coarse(df_a: pd.DataFrame, df_b: pd.DataFrame, x_prop, y_prop) -> float:
        """
        Find the intersect between a pair of DataFrames.
        This uses a crude approximation, it finds the index in both DataFrames where
        df_a[y_prop]-df_b[y_prop] is at it's smallest absolute value and then returns the
        x_prop value at this index.

        """

        index_cross = (df_a[y_prop] - df_b[y_prop]).abs().idxmin()

        x_a = df_a[x_prop].loc[index_cross]
        x_b = df_b[x_prop].loc[index_cross]

        if ~np.isclose(x_a, x_b):
            raise ValueError("Coordiates of bins different!")

        return x_a

    @staticmethod
    def interp(df_a: pd.DataFrame, df_b: pd.DataFrame, x_prop, y_prop, kind="linear"):
        x_a, y_a = df_a[x_prop], df_a[y_prop]
        x_b, y_b = df_b[x_prop], df_b[y_prop]
        if (x_a != x_b).all():
            raise ValueError("Grids not the same!")

        # interp1d(x_a, y_a, kind=kind)
        # interp1d(x_b, y_b, kind=kind)
        f_delta = interp1d(x_a, y_b - y_a, kind=kind, fill_value="extrapolate")
        # f_grad = interp1d(x_a,np.gradient(y_b-y_a,x_a),kind=kind) # interpolation of gradient
        # estimate from coarse!
        x_0 = PairIntersectFinders.coarse(df_a, df_b, x_prop, y_prop)
        dx = x_a.diff().mean()
        intersect = fsolve(
            f_delta,
            x0=x_0 - dx / 2,
        )

        if len(intersect) > 1:
            print("WARNING: Multiple intersects found")
            return intersect[0]
        elif len(intersect) == 1:
            return intersect[0]
        else:
            raise NotImplementedError("No intersect found")

    @staticmethod
    def spline(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        x_prop,
        y_prop,
        y_err=None,
        show_plots=False,
        s=0.5,
    ):
        x_a, y_a = df_a[x_prop], df_a[y_prop]
        x_b, y_b = df_b[x_prop], df_b[y_prop]
        if (x_a != x_b).all():
            raise ValueError("Grids not the same!")
        delta_y = y_a - y_b
        delta_not_na = ~delta_y.isna()
        delta_y = delta_y[delta_not_na]
        x = x_a[delta_not_na]

        if y_err is not None:
            y_err = 1 / (df_a[y_err] + df_b[y_err])[delta_not_na]  # add errors

        spline_delta = UnivariateSpline(x, delta_y, w=y_err, s=s)
        intersects = spline_delta.roots()
        if show_plots:
            print(intersects)
            plt.errorbar(x, delta_y, yerr=y_err, fmt="k.")
            plt.plot(x, spline_delta(x), c="blue", ls="--")
            for root in intersects:
                plt.axvline(root, ls="--", c="grey")
        if len(intersects) == 1:
            return intersects[0]
        else:
            raise NotImplementedError(
                "TODO: Raise warning if more than one intersect is found!"
            )


pair_intesect_methods = {
    "coarse": PairIntersectFinders.coarse,
    "interp": PairIntersectFinders.interp,
    "linear": partial(PairIntersectFinders.interp, kind="linear"),
    "cubic": partial(PairIntersectFinders.interp, kind="cubic"),
}


def multi_intersect(
    dfs: List[pd.DataFrame],
    x_prop,
    y_prop,
    intersect_method="linear",
    **intersect_kwargs,
) -> List[float]:
    """
    Insert a list of N DataFrames and find the (N-1) intersects of neighbouring data frames in the lists.

    dfs: List[pd.DataFrame] = List of DataFrames to find intersects of. Must be in the order they appear spatially

    x_prop: str = Name of x coordinate to find intersect in
    y_prop : str = Name of column to use to find intersections with.

    intersect_method:
        'interp' : interpolates the data sets, by default linearly
        'linear' : interpolates the data points linearly
        'cubic' : interpolates the data points cubically to find intersects.
        TODO: 'spline'
    """
    if intersect_method is None:
        intersect_finder = PairIntersectFinders.interp
    else:
        try:
            intersect_finder = pair_intesect_methods[intersect_method]
        except KeyError:
            raise NotImplementedError(f"Invalid Intersect Method! {intersect_method}")
    intersects = []
    # iterate over all pairs of DataFrames and fund the intersections
    for i, (df_a, df_b) in enumerate(zip(dfs, dfs[1:])):
        intersects.append(
            intersect_finder(
                df_a, df_b, y_prop=y_prop, x_prop=x_prop, **intersect_kwargs
            )
        )

    return intersects


# -> List[Tuple[Union(float, None), Union(float, None)]]
def bounds_from_intersects(
    intersects: List[float],
):
    """
    Find the limits either side of each dataframe.
    TODO: Check that the functions that calll this have the right signature
    """
    # N = len(dfs)
    # if N != len(intersects) + 1:
    #     raise ValueError("Must always be N-1 intersects!")
    N = len(intersects) + 1
    bounds = []

    for i in range(N):
        if i == 0:
            bounds.append((None, intersects[0]))
        elif i == N - 1:
            bounds.append((intersects[i - 1], None))
        else:
            bounds.append((intersects[i - 1], intersects[i]))
    if N != len(bounds):
        raise ValueError("There was not one set of bounds per DataFrame!")
    return bounds


def masks_from_intersects(
    dfs: List[pd.DataFrame],
    intersects: List[float],
    x_coord: str,
    padding: float = None,
) -> List[pd.DataFrame]:
    """
    From a list of intersects, create masks for the data

    Creates selections for each DataFrame that are either side of the intersect.
    Does the same work as bounds_from_intersects first, but then creates masks
    for each DataFrame.

    dfs: List[pd.DataFrame] = List of DataFrames to find intersects of. Must be in the order they appear spatially
    intersects: List[float] = List of intersects to create masks from
    x_coord: str = Name of x coordinate to find intersect in
    padding: float = Amount to pad the masks by. Default is 0.0

    """
    N = len(dfs)
    if padding is None:
        padding = 0.0
    if N != len(intersects) + 1:
        raise ValueError("Must always be N-1 intersects!")
    masks = []

    for i, df in enumerate(dfs):
        if i == 0:
            masks.append(df[x_coord] > intersects[0] - padding)
        elif i == N - 1:
            masks.append(df[x_coord] < intersects[i - 1] + padding)
        else:
            masks.append(
                (df[x_coord] < intersects[i - 1] + padding)
                | (df[x_coord] > intersects[i] - padding)
            )

    if N != len(masks):
        raise ValueError("There wasn't one mask per DataFrame!")

    return masks


def apply_masks(
    dfs: List[pd.DataFrame], masks: List[pd.DataFrame]
) -> List[pd.DataFrame]:
    return [df.mask(mask) for df, mask in zip(dfs, masks)]


def extrapolate_to_interface(
    df_a: pd.DataFrame,
    x_intersect: float,
    x_prop: str,
    fit_range=5,
    return_coeffs=True,
    return_fits=False,
    y_props=None,
    error_suffix=None,
    n=1,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Return a row extrapolated with a polynomial of order `n` to the interface, for a single DataFrame.

    df_a: pd.DataFrame = DataFrame to extrapolate to interface
    fit_range: float = Range either side of the interface to fit to.
    x_intersect: float = x coordinate of the interface
    x_prop: str = Name of x coordinate
    y_props: List[str] = List of columns to fit to. If None, fit to all columns.

    error_suffix : str = Suffix of error column to use for weighting. If None, no weighting is used.
                         TODO: Currently not working correctly


    returns a dataframe row with the order
    TODO: add weightings by errors
    TODO: Neaten weighting by errors code to handle missing columns.
    TODO: Weighting can only currently be used on columns when not None
    """

    df = df_a.query(f"{x_intersect-fit_range}<={x_prop}<={x_intersect+fit_range}")

    x = df[x_prop]
    # If intersect is a data point, return row with data in.
    if (x == x_intersect).any():
        return df[x == x_intersect]

    if y_props is None and error_suffix is not None:
        raise ValueError(
            "Cannot currently use weighted fitting without specifying columns"
        )

    print(y_props)

    if y_props is None:
        # Use all columns excluding x_prop and then readd
        y_props = list(df.drop(columns=x_prop).columns)
        # y = df #.drop(columns=x_prop) # Include x prop in fit to make life easier
    elif isinstance(y_props, str):
        y_props = [y_props]
        # Turn string into a list, so a DataFrame not a series is returned
        # y = df[[y_props]]
    y_props = y_props.copy()
    y_props.insert(0, x_prop)  # put x at the beginning of the list
    y = df[y_props]

    if error_suffix is None:
        # Use all columns
        weights = (
            None  # .drop(columns=x_prop) # Include x prop in fit to make life easier
        )
    else:
        # y_errs = [label+error_suffix for label in y_props]
        # weights = [1/df[y_errs]  ]
        all_cols = df.columns
        # Iterate over columns. If in all_cols set to 1/error, else just None
        weights = {
            err_col: 1 / df[err_col]
            if err_col in all_cols and err_col != x_prop
            else None
            for err_col in (col + error_suffix for col in y_props)
        }

    # Create polynomial fit object for each column
    if weights is None:
        # Unweighted fit for each column
        polys = {
            col: Polynomial.fit(x, y[col], n, w=None) for col in y_props
        }  # np.polynomial.polynomial.polyfit(x,y,deg=1)  #[::-1]
    else:
        # Weighted fit for each column
        polys = {
            col: Polynomial.fit(x, y[col], n, w=weights[col + error_suffix])
            for col in y_props
        }  # np.polynomial.polynomial.polyfit(x,y,deg=1)  #[::-1]
    # Find the value at the interface
    y_pred = {col: [poly(x_intersect)] for col, poly in polys.items()}
    # Convert prediction int DataFrame row
    df_pred = pd.DataFrame(y_pred)  # .reshape((1,-1)),columns=df.columns)

    # Return prediction and fit objects
    if return_fits:
        return df_pred, polys
    # Return prediction and coefficients
    elif return_coeffs:
        coeffs = {col: poly.coef for col, poly in polys.items()}
        return df_pred, coeffs  # fit
    # Just return prediction
    else:
        return df_pred


def interface_values(
    dfs: List[pd.DataFrame],
    bounds,
    x_prop,
    fit_range=5,
    y_props=None,
    n=1,
    return_fits=False,
    group_by_interface=False,
    error_suffix=None,
):
    """
    For all DataFrames in `dfs`, extrapolate to the interface defined by `bounds`.
    """
    # Return results grouped by the interface and not the DataFrame
    if group_by_interface:
        return interface_vals_by_interface(
            dfs,
            bounds,
            x_prop,
            fit_range=fit_range,
            y_props=y_props,
            n=n,
            return_fits=return_fits,
            error_suffix=error_suffix,
        )

    results = []
    fits = []
    for i, (bound, df) in enumerate(zip(bounds, dfs)):
        df_results = []
        df_fits = []
        for b in bound:
            if b is not None:
                interface_df, fit = extrapolate_to_interface(
                    df,
                    x_intersect=b,
                    x_prop=x_prop,
                    fit_range=fit_range,
                    return_coeffs=False,
                    return_fits=True,
                    y_props=y_props,
                    n=n,
                    error_suffix=error_suffix,
                )
                df_results.append(interface_df)
                if return_fits:
                    df_fits.append(fit)
        results.append(pd.concat(df_results))
        if return_fits:
            fits.append(df_fits)
    if return_fits:
        return results, fits
    return results


def interface_vals_by_interface(
    dfs: List[pd.DataFrame],
    bounds,
    x_prop,
    fit_range=5,
    y_props=None,
    n=1,
    return_fits=False,
    error_suffix=None,
):
    results = defaultdict(list)
    fits = []
    for i, (bound, df) in enumerate(zip(bounds, dfs)):
        for b in bound:
            if b is not None:
                interface_df, fit = extrapolate_to_interface(
                    df,
                    x_intersect=b,
                    x_prop=x_prop,
                    fit_range=fit_range,
                    return_coeffs=False,
                    return_fits=True,
                    y_props=y_props,
                    n=n,
                    error_suffix=error_suffix,
                )
                results[b].append(interface_df)
                fits.append(fit)
    results = pd.concat({key: pd.concat(dfs) for key, dfs in results.items()})

    if return_fits:
        return results, fits
    else:
        return results


@dataclass
class InterfaceFinder:
    """
    Find the interfaces between several DataFrames

    Create using:
    ```python
    finder = InterfaceFinder(dataframes, x_coord, y_coord)
    ```
    Access the interface locations using:
    ```python
    finder.intersects
    ```
    Extrapolate the properties to the interface using:
    ```python
    finder.interface_values()
    ```
    Attributes:
        Required:
        dataframes: List of DataFrames to find interfaces between
        x_coord: Name of the x coordinate column

        Optional:
        y_coord="density_number": Name of the y coordinate column to find the intersects between

        intersect_method: Method to use to find the intersects. See `multi_intersect` for options
        intersect_kwargs: Keyword arguments to pass to the intersect method
        pad_masks: Amount to pad the masks by. If None, no padding is applied
        intersects: List of the intersect locations, calculated in `__post_init__`
        bounds: List of the bounds of the masks, calculated in `__post_init__`
        masks: List of the masks, calculated in `__post_init__`
        masked_dfs: List of the masked DataFrames, calculated in `__post_init__`

    """

    dataframes: List[pd.DataFrame]

    x_coord: str
    y_coord: str = "density_number"

    intersect_method: str = "linear"
    intersect_kwargs: Dict = field(default_factory=dict)
    pad_masks: float = None

    intersects: List[float] = field(init=False)
    bounds: List[Tuple[float]] = field(init=False)
    masks: List[pd.DataFrame] = field(init=False)
    masked_dfs: List[pd.DataFrame] = field(init=False)

    def __post_init__(self):
        self.intersects = multi_intersect(
            self.dataframes,
            self.x_coord,
            self.y_coord,
            intersect_method=self.intersect_method,
            **self.intersect_kwargs,
        )
        self.bounds = bounds_from_intersects(self.intersects)
        self.masks = masks_from_intersects(
            self.dataframes, self.intersects, self.x_coord, padding=self.pad_masks
        )
        self.masked_dfs = apply_masks(self.dataframes, self.masks)

    def interface_values(
        self,
        fit_range=5,
        y_props=None,
        n=1,
        return_fits=False,
        group_by_interface=False,
        error_suffix=None,
    ):
        """
        Find the values extrapolated to the interfaces
        """
        return interface_values(
            self.masked_dfs,
            self.bounds,
            x_prop=self.x_coord,
            fit_range=fit_range,
            y_props=y_props,
            n=n,
            return_fits=return_fits,
            group_by_interface=group_by_interface,
            error_suffix=error_suffix,
        )

    def deltas(self, **interface_val_kwargs):
        # dfs = self.interface_values(**interface_val_kwargs)

        # df = pd.concat(dfs)
        # df.groupby(by = self.x_coord)

        deltas = {}

        for i, x in enumerate(self.intersects):
            y_a = extrapolate_to_interface(
                self.masked_dfs[i],
                x,
                self.x_coord,
                return_coeffs=False,
                **interface_val_kwargs,
            )
            y_b = extrapolate_to_interface(
                self.masked_dfs[i + 1],
                x,
                self.x_coord,
                return_coeffs=False,
                **interface_val_kwargs,
            )

            deltas[x] = y_a - y_b

        return pd.concat(deltas).droplevel(1)

    def make_plots(
        self,
        y_props,
        axs=None,
        show_extrap=False,
        show_original=False,
        extrap_options={"fit_range": 5, "n": 1},
        colours=None,
        **kwargs,
    ):
        import matplotlib.colors as mcolors

        if colours is None:
            colours = [c for c in mcolors.BASE_COLORS.values()]
        x_prop = self.x_coord
        dfs = self.masked_dfs
        og_dfs = self.dataframes
        x_inter = self.intersects
        if not show_extrap:
            interface_vals = self.interface_values(
                return_fits=show_extrap, y_props=y_props, **extrap_options
            )
        else:
            interface_vals, fits = self.interface_values(
                return_fits=show_extrap, y_props=y_props, **extrap_options
            )
        interface_vals_group = self.interface_values(
            group_by_interface=True, y_props=y_props, **extrap_options
        )

        fig, axs = plt.subplots(len(y_props), sharex=False)

        for ax, y_prop in zip(axs, y_props):
            for j, (df, og_df) in enumerate(zip(dfs, og_dfs)):
                # Plot the data
                if not show_original:
                    og_df = None
                make_plot(
                    df, x_prop, y_prop, ax, colour=colours[j], og_df=og_df, **kwargs
                )
                if show_extrap:
                    for k, x0 in enumerate(
                        (b for b in self.bounds[j] if b is not None)
                    ):
                        if x0 is not None:
                            xs = np.linspace(
                                x0 - extrap_options["fit_range"],
                                x0 + extrap_options["fit_range"],
                            )
                            plot_extrap(
                                df, y_prop, xs, fits[j][k], ax, colour=colours[j]
                            )

                x = interface_vals[j][x_prop]
                y = interface_vals[j][y_prop]
                ax.plot(x, y, marker=".", ls=" ", c=colours[j], mec="k")
            # plot the temperature jumps
            if y_prop != "density_number":
                for x, df in interface_vals_group.groupby(level=0):
                    y = df[y_prop]
                    ax.annotate(
                        "",
                        xy=(x, y.min()),
                        xytext=(x, y.max()),
                        xycoords="data",
                        textcoords="data",
                        arrowprops=dict(
                            arrowstyle="<->", connectionstyle="arc3", color="k", lw=1
                        ),
                    )

            for x in x_inter:
                ax.axvline(x, ls="--", c="grey")


def make_plot(df, x_prop, y_prop, ax, og_df=None, colour="red", **kwargs):
    ax.plot(df[x_prop], df[y_prop], c=colour, ls="-", **kwargs)
    if og_df is not None:
        ax.plot(og_df[x_prop], og_df[y_prop], c=colour, ls="-.", **kwargs)
    ax.set(xlabel=x_prop, ylabel=y_prop)


def plot_extrap(df, y_prop, xs, fits, ax, colour="red"):
    fit = fits[y_prop]
    ax.plot(xs, fit(xs), c=colour, ls="--")


def jumps():
    raise NotImplementedError("TODO: make a single plot")

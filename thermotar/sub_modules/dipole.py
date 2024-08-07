# import thermotar as th
from thermotar.utils import parse_logs
from thermotar.utils import df_utils
import thermotar.thermo as th
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class Dipoles(th.Thermo):

    """Class is essentially the Thermo class, with extra methods for computing dielectric properties
    The constructor takes an input dataframe and returns the
    """

    _tauD = None
    _diel = None

    # def __init__(self,df):

    @classmethod
    def from_xvg(cls, *files, **kwargs):
        # create a list of data frames
        dfs = [parse_logs.parse_xvg(file, **kwargs) for file in files]

        # reorder dfs by number of rows, so longest duplicated is kept
        # TODO this has issues when the dataframes columns with the same name start with different values, such as different times
        # need to work out a way of stictching this together

        # #dfs.sort(key=lambda x : len(x.index),reverse=True)

        # joined_df = pd.concat(dfs,axis=1)
        # duped_names = joined_df.columns.duplicated()

        # dfs_indexed

        # drop duplicate columns
        # joined_df = joined_df.loc[:,~joined_df.columns.duplicated()]
        joined_df = df_utils.merge_no_dupes(*dfs)
        return Dipoles(joined_df)

    @property
    def tauD(self):
        if self._tauD is None:
            return self.relaxation_time()[0]
        else:
            return self._tauD

    @property
    def diel(self):
        if self._diel is None:
            return self.calc_epsilon()
        else:
            return self._diel

    def relaxation_time(
        self,
        t_start=1,
        t_end=20,
        t_name="Time",
        corr_name="C",
        n_exp=1,
        method="fit",
        all_params=False,
        p0=None,
    ):
        """
        Find the debye relaxation time(s) (ps) and other fitting parameters
        """
        t = self.data[t_name]
        C = self.data[corr_name]

        def exp_func(t, a, tau):
            return a * np.exp(-t / tau)

        select = np.logical_and(t > t_start, t < t_end)

        if not p0:
            p0 = (
                1,
                t_end - t_start,
            )  # if p0 is not initialised, set the guess to be the range of the fit

        fit, cov = curve_fit(exp_func, t[select], C[select], p0=p0)

        self._tauD = fit[1]
        self.tau_fit = fit

        if all_params:
            return fit, cov
        else:
            # to do make it return error
            return fit[1], cov[1][1] ** 0.5

    def calc_epsilon(
        self,
        calc_method="auto",
        V=None,
        ave_lower=0.90,
        ave_upper=1.00,
        time_name="Time",
        epsilon_name="epsilon",
        M_fluc_name="Mfluc",
    ):
        """
        By default will calculate epsilon as the average of the last 10% of the epsilon column
        If epsilon is not defined as a dedicated column, can calculate from the fluctuation of the dipole moment
        can be manually specified
        """

        # todo: move to a library for reuse in other modules
        def average_range(df, ave_lower, ave_upper):
            ll = df[time_name].max() * ave_lower
            ul = df[time_name].max() * ave_upper
            select = np.logical_and(df[time_name] > ll, df[time_name] < ul)
            return df[epsilon_name][select].mean()

        if calc_method == "auto":
            try:
                eps = average_range(self.data, ave_lower, ave_upper)
            except IndexError:
                raise IndexError(f"No such column{epsilon_name}")

        return eps


if __name__ == "__main__":
    rep1 = Dipoles.from_xvg(
        "test_files/rep1/dipcorr.xvg", "test_files/rep1/epsilon.xvg"
    )
    tau = rep1.relaxation_time()[0]
    eps = rep1.diel
    eps = rep1.calc_epsilon()

    print(tau)
    print(eps)

    plt.plot(rep1.Time, rep1.C)
    plt.plot(rep1.Time, np.exp(-rep1.Time / tau))
    plt.yscale("log")
    plt.show()

    plt.plot(rep1.Time, rep1.epsilon)
    plt.axhline(eps, c="red")
    plt.show()

    print(rep1.data.head())

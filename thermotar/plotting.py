"""A collection of plotting functions I find myself writing often."""

from typing import Optional
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .thermo import Thermo


def shaded_errorbar(x, y, yerr, fmt="", alpha=0.5, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    y_above = y + yerr
    y_below = y - yerr

    if "color" in kwargs:
        color = kwargs.pop("color")
        # TODO: Also pop "c"
    elif "c" in kwargs:
        color = kwargs.pop("c")
    else:
        # TODO: get the color of the current axis
        color = None

    (line,) = ax.plot(x, y, fmt, color=color, **kwargs)  # type: ignore
    line_color = line.get_color()

    shading = ax.fill_between(
        x, y_below, y_above, color=line_color, alpha=alpha, **kwargs
    )

    return (line, shading)


def plot_thermostat_energies(
    thermo: Thermo,
    thermostat_H_label: str = "thermostatH",
    thermostat_C_label: str = "thermostatC",
) -> Figure:
    """Plot the average thermostat energies over the course of the simulation."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1: Axes
    ax1: Axes

    thermostatC = thermo[thermostat_C_label]
    thermostatH = thermo[thermostat_H_label]
    tot_eng_ave = thermo["TotEng"].mean()

    ax1.plot(thermo.Step, -thermostatH, label="- Hot Thermostat", c="red")
    ax1.plot(thermo.Step, thermostatC, label="Cold Thermostat", c="blue")
    ax1.set(ylabel=r"$\Delta E_\mathrm{thermo}$/ kcal/mol")
    ax1.legend()

    ax2.plot(
        thermo.Step,
        -(thermostatH + thermostatC),
        label="Hot + Cold Thermostat",
        c="black",
    )
    ax2.plot(
        thermo.Step,
        thermo.TotEng - tot_eng_ave,
        label="Total Energy - Exp[TotEng]",
        c="blue",
    )
    ax2.set(ylabel=r"$\delta E$/ kcal/mol")
    ax2.legend()

    ax3.plot(
        thermo.Step,
        -(thermostatH + thermostatC) / tot_eng_ave,
        label="Hot + Cold Thermostat",
        c="black",
    )
    ax3.plot(
        thermo.Step,
        (thermo.TotEng - tot_eng_ave) / tot_eng_ave,
        label="Total Energy",
        c="blue",
    )
    ax3.set(xlabel="Step", ylabel=r"$\delta E/ \langle E_tot \rangle$")
    ax3.legend()

    return fig

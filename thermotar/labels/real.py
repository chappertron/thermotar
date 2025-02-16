"""
A collection of commonly used labels for use in plotting.

This module exports labels with the LAMMPS real units.
"""

#: Axis label for the x-coordinate
X_COORD = r"$x/\ \mathrm{\AA}$"
#: Axis label for the y-coordinate
Y_COORD = r"$y/\ \mathrm{\AA}$"
#: Axis label for the z-coordinate
Z_COORD = r"$z/\ \mathrm{\AA}$"

DENS_MASS = r"$\rho_m/\ \mathrm{g\, cm^{-3}}$"
DENS_NUM = r"$\rho_N/\ \mathrm{\AA^{-3}}$"
DENS_CHARGE = r"$\rho_q/ e\mathrm{\AA^{-3}}$"
TEMP = r"$T/ \mathrm{K}$"

# Electrostatics
# These don't strictly have LAMMPS units defined in https://docs.lammps.org/units.html
ELECTRIC_FIELD = r"$E/\ \mathrm{V/\,\AA}$"
ELECTRIC_FIELD_Z = r"$E_z/\ \mathrm{V/\,\AA}$"
ELECTRIC_FIELD_VEC = r"$\mathbf{E}/\ \mathrm{V/\,\AA}$"
ELECTRIC_POT = r"$\phi/\ \mathrm{V}$"


# Temperature Gradients
TEMP_GRAD = r"$\nabla T / \ \mathrm{K/\,\AA}$"
TEMP_GRAD_Z = r"$\nabla_z T(z) / \ \mathrm{K/\,\AA}$"
TEMP_GRAD_VEC = r"$\mathbf{\nabla} T / \ \mathrm{K/\,\AA}$"

import thermotar as th
from thermotar.intersects import InterfaceFinder, multi_intersect, PairIntersectFinders
import matplotlib.pyplot as plt


def _is_zero_density(df):
    """
    Helper function that masking zero density.

    Using a helper allows us to load and mask in one line easily.
    """
    return df["density_number"] == 0.0


def _is_left(df):
    return df["Coord1"] < 0.0


# Load in the data for each section
# and mask where the densities are zero
# Masking ensures intersection isn't found where both have zero density
# Also masking the left interface, can only do one at a time.
df_au = (
    th.create_chunk("./resources/gold.chunk").data.mask(_is_zero_density).mask(_is_left)
)
df_ligand = (
    th.create_chunk("./resources/ligand.chunk")
    .data.mask(_is_zero_density)
    .mask(_is_left)
)
# For simplicity just using oxygen atoms.
# Ideally should use oxygen + hydrogen.
df_water = (
    th.create_chunk("./resources/oxygen.chunk")
    .data.mask(_is_zero_density)
    .mask(_is_left)
)


# Intersections
# Data frames must be in spatial order.
inter_finder = InterfaceFinder(
    [df_au, df_ligand, df_water], "Coord1", y_coord="density_number"
)

# Locations of the intersects
print(inter_finder.intersects)

# Masked Dataframes
print(inter_finder.masked_dfs)

# A method for plotting the data
# These plots aren't very pretty, but good for debugging
inter_finder.make_plots(["density_number", "temp"], show_original=True)
plt.show()

# Extrapolate the temperatures to the interfaces
print(inter_finder.interface_values(y_props="temp"))

# Estimate the temperature jumps at the interfaces
print(inter_finder.deltas(y_props="temp"))


# You can alternatively just get the interface locations using
intersects = multi_intersect([df_au, df_ligand, df_water], "Coord1", "density_number")

# And if you only care about one interface, you can use the `PairIntersect`s directly
PairIntersectFinders.interp(df_ligand, df_water, "Coord1", "density_number")

# utilities for cleaning up dataframes from lammps output

import re

# For renaming columns
#
#
# Striping prefixes from column names


def strip_pref(col_name: str) -> str:
    """Strip the LAMMPS prefixes from the provided column name."""
    # applied when read in
    # re for searching whether the df
    # for matching 'c_', 'f_' or 'v_' at the beginning of a string
    prefix_re = re.compile(r"^[cfv]_")

    col_stripped = prefix_re.sub("", col_name)

    return col_stripped


def drop_python_bad(col_name: str) -> str:
    """Remove characters from the column names (input string) that don't work in python variable names."""
    # applied when read in
    # re for searching whether the df
    # for matching 'c_', 'f_' or 'v_' at the beginning of a string

    left_bracket_re = re.compile(
        r"[<[{(]"
    )  # workout how to replace with _contents of this... # no backslashes needed because with in the group

    right_bracket_re = re.compile(
        r"[>})\]]"
    )  # workout how to replace with _contents of this... # no backslashes needed because with in the group

    non_python_re = re.compile(r"\W")  # matches any non word character

    col_name = left_bracket_re.sub("_", col_name)
    col_name = right_bracket_re.sub(
        "", col_name
    )  # replace with empty string so there is only an underscore before the thing, not before and after.

    col_stripped = non_python_re.sub(
        "_", col_name
    )  # replace any matched character with an underscore

    return col_stripped

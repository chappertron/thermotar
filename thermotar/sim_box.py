from dataclasses import dataclass
import re


@dataclass
class OrthogonalBox:
    """
    A class representing an orthogonal simulation box

    Fields
    ------

    xlo: float
        Lower bound in the x-axis
    xhi: float
        Upper bound in the x-axis
    ylo: float
        Lower bound in the y-axis
    yhi: float
        Upper bound in the y-axis
    zlo: float
        Lower bound in the z-axis
    zhi: float
        Upper bound in the z-axis

    """

    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float

    def lx(self) -> float:
        """
        The size of the box along x-axis
        """
        return self.xhi - self.xlo

    def ly(self) -> float:
        """
        The size of the box along y-axis
        """
        return self.yhi - self.ylo

    def lz(self) -> float:
        """
        The size of the box along z-axis
        """
        return self.zhi - self.zlo

    @staticmethod
    def from_lmp_data(file_name: str) -> "OrthogonalBox":
        """
        Read the box information directly from a LAMMPS data file

        Parameters
        ----------

        file_name: str
            The name of the file to read from.
        """

        REGEX = re.compile(r".+ xlo xhi")

        with open(file_name) as f:
            line = ""
            while REGEX.match(line) is None:
                line = f.readline()
            # A B xlo xhi
            (xlo, xhi) = line.split()[:2]
            line = f.readline()
            (ylo, yhi) = line.split()[:2]
            line = f.readline()
            (zlo, zhi) = line.split()[:2]

        return OrthogonalBox(
            xlo=float(xlo),
            xhi=float(xhi),
            ylo=float(ylo),
            yhi=float(yhi),
            zlo=float(zlo),
            zhi=float(zhi),
        )

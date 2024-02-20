### performance testing
import thermotar as th


def test_index():
    thermo = th.create_thermos("./tests/test_files/log.lammps")

    assert (thermo['Temp'] == thermo.data['Temp']).all()

import pytest
from thermotar.utils import parse_logs


def test_get_lmp_properties():
    file = './thermotar/tests/test_files/nemd_rtp6.log'
    
    print(parse_logs.get_lmp_properties(file))


if __name__ == "__main__":

    test_get_lmp_properties()
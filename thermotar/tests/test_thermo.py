### performance testing

import cProfile
import pstats

import thermotar as th


def parse_perf_test():
    THERMO = th.Thermo.create_thermos("test_files/hexane15.log")

    return THERMO


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        parse_perf_test()

    stats = pstats.Stats(pr)
    stats.dump_stats("./logs/perf.prof")

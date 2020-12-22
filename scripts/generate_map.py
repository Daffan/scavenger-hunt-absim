import sys
import argparse
from os.path import join, abspath, dirname
sys.path.append(dirname(dirname(join(abspath(__file__)))))
from map import generate

NODES_RANGE = [8, 8]
COST_RANGE = [10, 500]
OBJECTS_RANGE = [4, 4]
OCCURRENCES_RANGE = [1, 4]

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--map", help="path to the map file", type=str, default="maps/default.dat"
    )
    fname = parser.parse_args().map
    generate(fname, NODES_RANGE, COST_RANGE, OBJECTS_RANGE, OCCURRENCES_RANGE)

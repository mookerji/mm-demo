#! /usr/bin/env python3

# Implements:
# - https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/

import click
import json
import numpy as np
import osmnx as nx
import pandas as pd

import logging
import os

import graphical_models as gm
import spatial
import traj as traj

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)


@click.command()
@click.option('--input-filename', default='san-francisco.graphml')
@click.option('--input-trace', default='trace.txt')
@click.option('--num-points', type=int, default=-1)
def main(input_filename, input_trace, num_points):
    network = spatial.RoadNetwork.from_file(input_filename)
    trajectory = traj.Trajectory.from_file(input_trace)
    if num_points > 0:
        trajectory = trajectory.head(num_points)
    hmm = gm.HiddenMarkovModel(trajectory, network).init_model()
    path = hmm.decode()
    logging.debug('trellis=%d', len(hmm._trellis))
    print(json.dumps(path.to_geojson()))


if __name__ == '__main__':
    main()

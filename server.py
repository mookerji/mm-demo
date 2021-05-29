import json
import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS

import graphical_models as gm
import mbxapi
import spatial
import traj as traj

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)

app = Flask(__name__)
CORS(app)

network = spatial.RoadNetwork.from_file('san-francisco.graphml')


@app.route('/healthz')
def healthz():
    return 'Hello, World!'


def parse_coordinates(cs):
    coordinates = [pair.split(',') for pair in cs.split(';')]
    coordinates_ = [{
        'longitude': float(lng),
        'latitude': float(lat)
    } for (lng, lat) in coordinates]
    return traj.Trajectory.from_points(coordinates_)


def handle_coordinates(coordinates):
    trajectory = parse_coordinates(coordinates)
    hmm = gm.HiddenMarkovModel(trajectory, network).init_model()
    path = hmm.decode()
    print(json.dumps(path.to_geojson()))
    print(mbxapi.viterbi_path_to_mbx(path).to_json())
    return jsonify(mbxapi.viterbi_path_to_mbx(path).to_json())


@app.route('/matching/v5/mapbox/<profile>/<coordinates>')
def matching1(profile, coordinates):
    return handle_coordinates(coordinates)


@app.route('/matching/v5/mapbox/<profile>/<coordinates>.json')
def matching(profile, coordinates):
    return handle_coordinates(coordinates)


if __name__ == '__main__':
    app.run(threaded=True, port=8080)

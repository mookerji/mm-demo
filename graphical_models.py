"""
Graphical models
"""

from shapely.geometry import LineString

import attr
import geopandas as gp
import networkx as nx
import numpy as np
import pandas as pd

from observability import observer
import spatial

DEFAULT_SIGMA_Z = 5
DEFAULT_BETA = 3

# Probability functions


@attr.s
class EmissionLikelihood(object):

    sigma_z = attr.ib(default=DEFAULT_SIGMA_Z)

    def __call__(self, sample_pos, road_edge):
        road_pos = spatial.get_nearest_point(road_edge.geometry, sample_pos)
        l2_distance_ = spatial.geodetic_distance_meters(sample_pos, road_pos)
        return -0.5 * (l2_distance_ / self.sigma_z)**2 - 0.5 * np.log(
            2 * np.pi * self.sigma_z**2)


# TODO(mookerji): Find out if this transition probability model differs from
# what's in Valhalla / OSRM.
@attr.s
class TransitionLikelihood(object):

    network = attr.ib()
    beta = attr.ib(default=DEFAULT_BETA)

    def __call__(self, src, src_edge, dst, dst_edge):
        l2_distance_ = spatial.geodetic_distance_meters(src, dst)
        l1_distance_ \
          = self.network.get_network_distance_meters(src, src_edge, dst, dst_edge)
        d = np.abs(l2_distance_ - l1_distance_)
        return -d / self.beta - np.log(self.beta)


## Graphs

STATE_START = -1
STATE_END = -2


@attr.s(frozen=True)
class Tracepoint(object):
    """
    Matches trajectory measurement to matching point on the road network.
    """

    location = attr.ib(default=None)
    distance = attr.ib(default=None)
    name = attr.ib(default=None)
    waypoint_index = attr.ib(default=0)
    matchings_index = attr.ib(default=0)
    alternatives_count = attr.ib(default=0)


@attr.s(eq=True, frozen=True)
class StateNode(object):
    """
    State of a Viterbi path
    """

    trace_id = attr.ib()
    edge_id = attr.ib()
    is_virtual = attr.ib(default=False, repr=False)

    def __eq__(self, o):
        return self.trace_id == o.trace_id and self.edge_id == o.edge_id

    def __neq__(self, o):
        return not self.__eq__(o)


@attr.s
class ViterbiPath(object):

    path = attr.ib(default=[])
    model = attr.ib(default=None)

    def get_geometry(self):
        path_edges = set(
            [leg.edge_id for leg in self.path if not leg.is_virtual])
        # TODO(mookerji): This is a serious hack. Ideally, we'd have a
        # traversal of all the nodes, where we fill in any intermediate paths,
        # and then build the geometry off of that.
        # TODO(mookerji): Handle any truncation/trimming
        all_points = []
        for edge in path_edges:
            geometry = self.model.network.get_edge_by_id(edge).geometry
            all_points.extend([c for c in geometry.coords])
        return LineString(all_points)

    def get_tracepoints(self):
        tracepoints = []
        for leg in self.path:
            if leg.is_virtual:
                continue
            edge = self.model.network.get_edge_by_id(leg.edge_id)
            point = self.model.measurements.get_trace_id(leg.trace_id).geometry
            nearest = spatial.get_nearest_point(edge.geometry, point)
            tracepoints.append(
                Tracepoint(
                    location=[point.x, point.y],
                    distance=spatial.geodetic_distance_meters(point, nearest),
                    name=edge['name'],
                    waypoint_index=leg.trace_id,
                ))
        return tracepoints

    def get_distance_meters(self):
        # TODO(mookerji): Implement, Handle any truncation/trimming
        raise NotImplementedError()

    def get_duration_sec(self):
        # TODO(mookerji): Implement, Handle any truncation/trimming
        raise NotImplementedError()

    def get_weight(self):
        # TODO(mookerji): Implement, Handle any truncation/trimming
        raise NotImplementedError()

    def to_geojson(self):
        path_edges = set(
            [leg.edge_id for leg in self.path if not leg.is_virtual])
        return spatial.tjs(
            pd.concat([
                self.model.network.edge_table.loc[path_edges].geometry,
                self.model.measurements.data.geometry,
            ]))

    def get_score(self):
        score = 0
        prev = None
        for node in path:
            if not prev:
                prev = node
                continue
            score += self.model.get_edge_likelihood(prev, node)
            prev = node
        return score


@attr.s
class HiddenMarkovModel(object):
    """
    Batch-oriented HMM for estimating most likely sequence of
    edges. Takes as a prior a road network, series of measurements
    (i.e., trajectory), and measurement/transition models.
    """

    # Priors
    measurements = attr.ib()
    network = attr.ib()
    emission_likelihood = attr.ib(init=False)
    transition_likelihood = attr.ib(init=False)

    # Internal state around the graphical model
    _trellis = attr.ib(default=None)
    _trellis_start = attr.ib(default=None)
    _trellis_end = attr.ib(default=None)

    def __attrs_post_init__(self):
        # TODO(mookerji): Maybe these should be passed in?
        self.emission_likelihood = EmissionLikelihood()
        self.transition_likelihood = TransitionLikelihood(self.network)

    def is_initialized(self):
        return all([self._trellis, self._trellis_start, self._trellis_end])

    def init_model(self):
        # Steps:
        # - For each trace point get a list of edges
        # - Create adjacency list: each edge can tragnsition to every other edge
        # - Decode Djikstra search on the adjacency list
        self._trellis = nx.DiGraph()
        self._trellis_start \
          = StateNode(STATE_START, STATE_START, is_virtual=True)
        self._trellis.add_node(self._trellis_start)
        prev_layer = [self._trellis_start]

        # Iterate through measurements
        for trace_index, trace_item in self.measurements.data.iterrows():
            edges = self.network.nearest_edges(trace_item.geometry)
            current_layer = []
            for edge_index, edge in edges.iterrows():
                node = StateNode(trace_index, edge_index)
                self._trellis.add_node(node)
                for prev in prev_layer:
                    self._trellis.add_edge(prev, node)
                current_layer.append(node)
            prev_layer = current_layer

        # Add final node
        self._trellis_end = StateNode(STATE_END, STATE_END, is_virtual=True)
        self._trellis.add_node(self._trellis_end)
        for prev in prev_layer:
            self._trellis.add_edge(prev, self._trellis_end)
        return self

    def decode(self):
        assert self.is_initialized()
        return ViterbiPath(
            nx.dijkstra_path(
                self._trellis,
                self._trellis_start,
                self._trellis_end,
                weight=self.get_edge_likelihood,
            ), self)

    @observer()
    def get_edge_likelihood(self, start_node, end_node, attributes={}):
        # TODO(mookerji): Turn this into a decorator
        #log.debug('get-edge-likelihood', start=start_node, end=end_node)
        if start_node.is_virtual:
            return 1
        if end_node.is_virtual:
            start = self.measurements.data.loc[start_node.trace_id]
            start_edge = self.network.edge_table.loc[start_node.edge_id]
            # See comment below about negative weights
            return -self.emission_likelihood(start.geometry, start_edge)

        start = self.measurements.get_trace_id(start_node.trace_id)
        start_edge = self.network.get_edge_by_id(start_node.edge_id)
        end = self.measurements.get_trace_id(end_node.trace_id)
        end_edge = self.network.get_edge_by_id(end_node.edge_id)
        emission_weight = self.emission_likelihood(start.geometry, start_edge)
        transition_weight = self.transition_likelihood(
            start.geometry,
            start_edge,
            end.geometry,
            end_edge,
        )
        # Use negative weight so that the log-likelihoods are non-negative for
        # graph search
        return -transition_weight + -emission_weight

    def draw(self):
        #nx.nx_pydot.write_dot(trellis, input_trajectory + '.dot')
        raise NotImplementedError()

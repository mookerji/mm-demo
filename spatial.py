"""
Geometry and spatial query functions

## What the tables/key data models look like:

Graph (GeoPandas DataFrame) - The Road Network

           u           v  key      osmid     highway  oneway  length                                           geometry                             name
8   65323022  4417156845    0  162799068  [tertiary]   False  36.385  LINESTRING (-122.46651 37.76704, -122.46654 37...  Martin Luther King Junior Drive
9   65323022  4417156840    0  162799068  [tertiary]   False  17.895  LINESTRING (-122.46651 37.76704, -122.46650 37...  Martin Luther King Junior Drive
10  65323030  4025608217    0   32928798  [tertiary]   False   9.866  LINESTRING (-122.46699 37.76802, -122.46708 37...  Martin Luther King Junior Drive
11  65323030  1211324459    0   30897687  [tertiary]   False  10.224  LINESTRING (-122.46699 37.76802, -122.46698 37...               Nancy Pelosi Drive
12  65323030  1211324558    0  162799068  [tertiary]   False  10.817  LINESTRING (-122.46699 37.76802, -122.46695 37...  Martin Luther King Junior Drive

Edges - Candidate road network edges returned from spatial queries

              u         v  key      osmid        highway  oneway   length                                           geometry           name
index
3082   65302780  65302782    0  254759977  [residential]   False  207.460  LINESTRING (-122.46403 37.77723, -122.46402 37...     6th Avenue
3086   65302782  65302780    0  254759977  [residential]   False  207.460  LINESTRING (-122.46389 37.77537, -122.46390 37...     6th Avenue
6466   65314570  65302780    0   86331217     [tertiary]   False   94.204  LINESTRING (-122.46510 37.77719, -122.46426 37...  Balboa Street
3080   65302780  65314570    0   86331217     [tertiary]   False   94.204  LINESTRING (-122.46403 37.77723, -122.46426 37...  Balboa Street
3079   65302780  65302778    0  158468970  [residential]   False  208.339  LINESTRING (-122.46403 37.77723, -122.46416 37...     6th Avenue
"""

from functools import partial

import logging
import os

from geopy.distance import lonlat, distance
from shapely.ops import transform
from sklearn.neighbors import BallTree

import attr
import cachetools
import geopandas as gp
import networkx as nx
import numpy as np
import osmnx as osm
import pandas as pd
import pyproj

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)

projectWGS84 = partial(
    pyproj.transform,
    pyproj.Proj(init='EPSG:4326'),
    pyproj.Proj(init='EPSG:32633'),
)

DEFAULT_SEARCH_RADIUS = 0.0005  # Deg


def geodetic_distance_meters(p1, p2):
    """
    Calculates the geodetic/WGS84 distance between two points in meters.
    """
    return distance(lonlat(*p1.coords[:][0]), lonlat(*p2.coords[:][0])).meters


GEOMETRY_CACHE = cachetools.LRUCache(maxsize=2**16)


# BUG: This is ... incorrect?
@cachetools.cached(GEOMETRY_CACHE, key=str)
def get_geometry_length_meters(line):
    return transform(projectWGS84, line).length


def is_same_edge(edge1, edge2):
    return edge1.u == edge2.u and edge1.v == edge2.v


# TODO(mookerji): check that this matches shapely.ops.nearest_points
# Arguments: shapely
def get_nearest_point(edge, point):
    """
    Gets the nearest point on an edge to a given point.
    """
    return edge.interpolate(edge.project(point))


## Distance functions


def network_key(src, src_edge, dst, dst_edge, network):
    return hash(
        frozenset({
            'src.x': src.x,
            'src.y': src.y,
            'src_edge.u': src_edge.u,
            'src_edge.v': src_edge.v,
            'dst.x': dst.x,
            'dst.y': dst.y,
            'dst_edge.u': dst_edge.u,
            'dst_edge.v': dst_edge.v,
        }))


# BUG(mookerji): Caching introduces errors?
@cachetools.cached(GEOMETRY_CACHE, key=network_key)
def get_network_distance_meters(src, src_edge, dst, dst_edge, network):
    """
    Computes the network distance between two points, where those
    points have been projected to road edges
    """
    # Base case: on the same edge
    src_pct = src_edge.geometry.project(src, normalized=True)
    dst_pct = dst_edge.geometry.project(dst, normalized=True)
    if is_same_edge(src_edge, dst_edge):
        return np.abs(src_pct - dst_pct) * src_edge.length

    # More elaborate network distance case: project measurements to edges and
    # then network distance computed from projected points
    src_projected = get_nearest_point(src_edge.geometry, src)
    src_node, src_dist = osm.get_nearest_node(
        network.graph,
        (src_projected.y, src_projected.x),
        return_dist=True,
    )
    dst_projected = get_nearest_point(dst_edge.geometry, dst)
    dst_node, dst_dist = osm.get_nearest_node(
        network.graph,
        (dst_projected.y, dst_projected.x),
        return_dist=True,
    )
    assert src_dist >= 0 and dst_dist >= 0

    route = nx.shortest_path(network.graph, src_node, dst_node)
    if LOG_LEVEL == 'DEBUG' and False:
        fig, ax = osm.plot_graph_route(
            network.graph,
            route,
            origin_point=(src_projected.y, src_projected.x),
            destination_point=(dst_projected.y, dst_projected.x),
            save=True,
            show=False,
            filename='%d-to-%d' % (src_node, dst_node),
        )
    assert route

    # In the trivial case where both the source and the destination share a
    # common node, then the total network distance is the sum of the distances
    # from each node.
    if len(route) == 1:
        return src_dist + dst_dist

    # Get network distance of route
    distance_ = 0
    current_node = src_edge.u
    for node in route[1:]:
        distance_ += network.get_distance_by_node(current_node, node)
        current_node = node

    assert len(route) >= 2
    # ... adjust networkx distance of the route depending on whether the
    # projected point intersects with the first (or last) edge of the route
    route_start_edge = network.get_geometry_by_node(route[0], route[1])
    if src.intersects(route_start_edge):
        distance_ -= src_dist
    else:
        distance_ += src_dist
    route_end_edge = network.get_geometry_by_node(route[-1], route[-2])
    if dst.intersects(route_end_edge):
        distance_ -= dst_dist
    else:
        distance_ += dst_dist
    return min(distance_, 0)


# Initialization lifted from:
# https://github.com/gboeing/osmnx/blob/5c7233743dd2e65024f5764fba745371d82e4b39/osmnx/geo_utils.py#L436
@attr.s
class SpatialEdgeIndex(object):

    point_index_ = attr.ib(default=None)
    extended_ = attr.ib(default=None, repr=False)
    graph_ = attr.ib(default=None, repr=False)

    @classmethod
    def from_graph(cls, graph):
        """
        Create a SpatialEdgeIndex from a NetworkX graph
        """
        graph = graph.copy()
        dist = 0.0001
        graph['points'] \
          = graph.apply(lambda x: osm.redistribute_vertices(x.geometry, dist), axis=1)
        #import pdb; pdb.set_trace()
        extended = graph['points'].apply([pd.Series]).stack().reset_index(
            level=1, drop=True).join(graph).reset_index()
        nodes = pd.DataFrame({
            'x': extended['Series'].apply(lambda x: x.x),
            'y': extended['Series'].apply(lambda x: x.y)
        })
        point_index = BallTree(nodes, metric='haversine')
        return cls(point_index, extended, graph)

    def query_edges(self, point, radius=DEFAULT_SEARCH_RADIUS):
        """
        Spatial query of nearest edges: given a shapely point will
        return all edges (as a GeoPandas DataFrame) within a radius.
        """
        index = self.point_index_.query_radius([[point.x, point.y]], radius)
        return self.extended_.loc[index[0]].drop_duplicates('index').set_index(
            'index').drop(labels=['Series', 'points'], axis=1)


LABELS_TO_DROP = [
    'access',
    'bridge',
    'junction',
    'lanes',
    'maxspeed',
    'ref',
    'service',
    'tunnel',
    'width',
]

# Non-driving related tags to filter
IGNORE_TAGS = set([
    'footway',
    'cycleway',
    'path',
    'pedestrian',
    'steps',
])


@attr.s
class RoadNetwork(object):
    """
    Queryable container for the road network.
    """

    graph = attr.ib()
    edge_table = attr.ib()
    edge_index = attr.ib()

    @classmethod
    def from_file(cls, filename):
        graph = osm.load_graphml(filename)
        nodes, edges = osm.graph_to_gdfs(
            graph,
            nodes=True,
            edges=True,
            node_geometry=True,
            fill_edge_geometry=True,
        )
        convert = lambda t: t if isinstance(t, list) else [t]
        edges['highway'] = edges.highway.apply(convert)
        filter_mask = edges.highway.apply(
            lambda t: len(set(IGNORE_TAGS) & set(t)) > 0)
        edges = edges[~filter_mask].drop(
            labels=LABELS_TO_DROP,
            errors='ignore',
            axis=1,
        )
        graph = osm.gdfs_to_graph(nodes, edges)
        index = SpatialEdgeIndex.from_graph(edges)
        return RoadNetwork(graph, edges, index)

    def get_network_distance_meters(self, src, src_edge, dst, dst_edge):
        return get_network_distance_meters(src, src_edge, dst, dst_edge, self)

    def nearest_edges(self, point, radius=DEFAULT_SEARCH_RADIUS):
        return self.edge_index.query_edges(point, radius)

    def get_edge_by_id(self, edge_id):
        return self.edge_table.loc[edge_id]

    def get_geometry_by_node(self, u, v):
        table = self.edge_table
        return table[(table.u == u) & (table.v == v)].geometry.iloc[0]

    def get_distance_by_node(self, u, v):
        # TODO(mookerji): There's a bug here somewhere!
        try:
            assert len(self.graph[u][v]) == 1
        except:
            return np.inf
        return self.graph[u][v][0]['length']


def tjs(geometry):
    return gp.GeoSeries(geometry).__geo_interface__

"""
Utilities for serializing to the Mapbox API
"""

import attr

import spatial

@attr.s
class Matching(object):

    geometry = attr.ib(default=1)
    confidence = attr.ib(default=1)
    legs = attr.ib(default=[])
    weight_name = attr.ib(default='routability')
    weight = attr.ib(default=None)
    duration = attr.ib(default=22)
    distance = attr.ib(default=222)


@attr.s
class MapboxResponse(object):

    matchings = attr.ib(default=[])
    tracepoints = attr.ib(default=[])
    code = attr.ib(default='Ok')

    def to_json(self):
        return attr.asdict(self)


def viterbi_path_to_mbx(path):
    geometry = spatial.tjs(path.get_geometry())['features'][0]['geometry']
    matching = Matching(geometry)
    return MapboxResponse([matching], path.get_tracepoints())

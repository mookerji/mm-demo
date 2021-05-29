import attr
import geopandas as gp
import pandas as pd

@attr.s
class Trajectory(object):
    """
    Container class for trajectory; wraps a GeoPandas DataFrame.

    Trajectory.data has shape:

        longitude   latitude                     geometry
    0 -122.462626  37.777296  POINT (-122.46263 37.77730)
    1 -122.461960  37.777313  POINT (-122.46196 37.77731)
    2 -122.461252  37.777346  POINT (-122.46125 37.77735)
    3 -122.460694  37.777397  POINT (-122.46069 37.77740)
    4 -122.460399  37.777406  POINT (-122.46040 37.77741)
    """

    data = attr.ib(default=None)

    @classmethod
    def from_points(cls, points):
        df = pd.DataFrame(points)
        geometry = gp.points_from_xy(df.longitude, df.latitude)
        return Trajectory(gp.GeoDataFrame(df, geometry=geometry))

    @classmethod
    def _from_csv(cls, filename):
        df = pd.read_csv(filename)
        geometry = gp.points_from_xy(df.longitude, df.latitude)
        return Trajectory(gp.GeoDataFrame(df, geometry=geometry))

    @classmethod
    def _from_geojson(cls, filename):
        df = gp.read_file(filename)
        df.longitude = df.geometry.x
        df.latitude = df.geometry.y
        return Trajectory(data=df)

    @classmethod
    def from_file(cls, filename):
        is_geojson = 'json' in filename
        if is_geojson:
            return Trajectory._from_geojson(filename)
        return Trajectory._from_csv(filename)

    def head(self, count):
        self.data = self.data.head(count)
        return self

    def to_geojson(self):
        return self.data.geometry.__geo_interface__

    def get_trace_id(self, trace_id):
        return self.data.loc[trace_id]

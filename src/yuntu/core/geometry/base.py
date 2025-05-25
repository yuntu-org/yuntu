"""Yuntu Geometries Module.

This module defines the base Geometries to be used for all
Yuntu objects. A geometry is a region of coordinate space with
time and/or frequency as axis.
"""
from abc import ABC
from abc import abstractmethod
from enum import Enum

import yuntu.core.utils.atlas as utils
import yuntu.core.windows as windows
import shapely.wkt

INFINITY = 10e+15


class Geometry(ABC):
    name = None

    class Types(Enum):
        Weak = 'Weak'
        TimeLine = 'TimeLine'
        TimeInterval = 'TimeInterval'
        FrequencyLine = 'FrequencyLine'
        FrequencyInterval = 'FrequencyInterval'
        BBox = 'BBox'
        Point = 'Point'
        LineString = 'LineString'
        Polygon = 'Polygon'
        MultiPoint = 'MultiPoint'
        MultiLineString = 'MultiLineString'
        MultiPolygon = 'MultiPolygon'
        GeometryCollection = 'GeometryCollection'

    def __init__(self, geometry=None):
        self.geometry = geometry

    def __repr__(self):
        args = ', '.join([
            '{}={}'.format(key, repr(value))
            for key, value in self.to_dict().items()
            if key != 'type'
        ])
        name = type(self).__name__
        return f'{name}({args})'

    @property
    def window(self):
        left, bottom, right, top = self.bounds

        if left is None or right is None:
            return windows.FrequencyWindow(min=bottom, max=top)

        if bottom is None or top is None:
            return windows.TimeWindow(start=left, end=right)

        return windows.TimeFrequencyWindow(
            min=bottom, max=top, start=left, end=right)

    def buffer(self, buffer=None, **kwargs):
        # pylint: disable=import-outside-toplevel
        import yuntu.core.geometry.polygons as polygons

        time, freq = utils._parse_tf(buffer, 'buffer', **kwargs)
        buffer_geom = utils.buffer_geometry(
            self.geometry,
            buffer=[time, freq])

        if buffer_geom.geom_type == 'Polygon':
            return polygons.Polygon(geometry=buffer_geom)

        return polygons.MultiPolygon(geometry=buffer_geom)

    def shift(self, shift=None, **kwargs):
        time, freq = utils._parse_tf(shift, 'shift', **kwargs)
        translated = utils.translate_geometry(
            self.geometry,
            xoff=time,
            yoff=freq)
        return type(self)(geometry=translated)

    def scale(self, scale=None, center=None, **kwargs):
        # pylint: disable=import-outside-toplevel
        import yuntu.core.geometry.points as points

        time, freq = utils._parse_tf(scale, 'scale', default=1, **kwargs)

        if center is None:
            center = 'center'

        if isinstance(center, points.Point):
            center = center.geometry

        scaled = utils.scale_geometry(
            self.geometry,
            xfact=time,
            yfact=freq,
            origin=center)
        return type(self)(geometry=scaled)

    def rotate(self, angle, origin='center', use_radians=True):
        # pylint: disable=import-outside-toplevel
        import yuntu.core.geometry.points as points

        if isinstance(origin, points.Point):
            origin = origin.geometry

        rotated = utils.rotate_geometry(
            self.geometry,
            angle,
            origin=origin,
            use_radians=use_radians)
        return type(self)(geometry=rotated)

    def transform(self, transform):
        transformed = utils.transform_geometry(
            self.geometry,
            transform)
        return type(self)(geometry=transformed)

    def intersects(self, other):
        if isinstance(other, (windows.Window, Geometry)):
            return self.geometry.intersects(other.geometry)

        return self.geometry.intersects(other)

    def intersection(self, other):
        if isinstance(other, (windows.Window, Geometry)):
            other = other.geometry

        new_geometry = self.geometry.intersection(other)
        return self.from_geometry(new_geometry)

    def union(self, other):
        # pylint: disable=import-outside-toplevel
        import yuntu.core.geometry.geometry_collections as geometry_collections
        import yuntu.core.geometry.intervals as intervals
        import yuntu.core.geometry.lines as lines

        if isinstance(other, (windows.Window, Geometry)):
            other = other.geometry

        if isinstance(other, (
                lines.TimeLine,
                lines.FrequencyLine,
                intervals.TimeInterval,
                intervals.FrequencyInterval,
                geometry_collections.GeometryCollection)):
            return geometry_collections.GeometryCollection([self, other])

        new_geometry = self.geometry.union(other)
        return self.from_geometry(new_geometry)

    @abstractmethod
    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get('figsize', None))

        return ax

    @property
    def bounds(self):
        return self.geometry.bounds

    def to_dict(self):
        return {'type': self.name.value}

    @staticmethod
    def from_geometry(geometry):
        geom_type = Geometry.Types[geometry.geom_type]
        geom_class = Geometry.get_class_from_name(geom_type)
        return geom_class(geometry=geometry)

    @staticmethod
    def from_dict(data):
        data = data.copy()
        geom_type = Geometry.Types[data.pop('type')]
        geom_class = Geometry.get_class_from_name(geom_type)

        if Geometry.Types.TimeInterval == geom_class.name:
            if "wkt" in data:
                if "start_time" not in data or "end_time" not in data:
                    start_time, _, end_time, _ = shapely.wkt.loads(data["wkt"]).bounds
                    data["start_time"] = start_time
                    data["end_time"] = end_time
                del data["wkt"]
        elif Geometry.Types.BBox == geom_class.name:
            if "wkt" in data:
                if "start_time" not in data or "end_time" not in data or "max_freq" not in data or "min_freq" not in data:
                    start_time, min_freq, end_time, max_freq= shapely.wkt.loads(data["wkt"]).bounds
                    data["start_time"] = start_time
                    data["end_time"] = end_time
                    data["min_freq"] = min_freq
                    data["max_freq"] = max_freq
                del data["wkt"]
        elif Geometry.Types.Weak == geom_class.name:
            del data["wkt"]

        return geom_class(**data)

    # pylint: disable=too-many-return-statements
    @staticmethod
    def get_class_from_name(name):
        # pylint: disable=import-outside-toplevel
        import yuntu.core.geometry.bbox as bbox
        import yuntu.core.geometry.geometry_collections as geometry_collections
        import yuntu.core.geometry.intervals as intervals
        import yuntu.core.geometry.lines as lines
        import yuntu.core.geometry.linestrings as linestrings
        import yuntu.core.geometry.points as points
        import yuntu.core.geometry.polygons as polygons
        import yuntu.core.geometry.weak as weak

        if name == Geometry.Types.Point:
            return points.Point

        if name == Geometry.Types.LineString:
            return linestrings.LineString

        if name == Geometry.Types.Polygon:
            return polygons.Polygon

        if name == Geometry.Types.MultiPoint:
            return points.MultiPoint

        if name == Geometry.Types.MultiLineString:
            return linestrings.MultiLineString

        if name == Geometry.Types.MultiPolygon:
            return polygons.MultiPolygon

        if name == Geometry.Types.GeometryCollection:
            return geometry_collections.GeometryCollection

        if name == Geometry.Types.Weak:
            return weak.Weak

        if name == Geometry.Types.TimeLine:
            return lines.TimeLine

        if name == Geometry.Types.FrequencyLine:
            return lines.FrequencyLine

        if name == Geometry.Types.TimeInterval:
            return intervals.TimeInterval

        if name == Geometry.Types.FrequencyInterval:
            return intervals.FrequencyInterval

        if name == Geometry.Types.BBox:
            return bbox.BBox

        message = f'Geometry Type {name} has not been implemented'
        raise NotImplementedError(message)

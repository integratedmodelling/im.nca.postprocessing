"""
Utility functions for geospatial data analysis.
"""

import numpy as np
import math
from shapely.geometry import Polygon
import geopandas as gpd


def area_of_pixel(pixel_size, center_lat):
    """
    area_of_pixel calculates the area, in hectares, of a wgs84 square raster
    tile given its latitude and side-length.
    This function is adapted from https://gis.stackexchange.com/a/288034.

    :param pixel_size: is the length of the pixel side in degrees.
    :param center_lat: is the latitude of the center of the pixel. This value
    +/- half the `pixel-size` must not exceed 90/-90 degrees latitude or an
    invalid area will be calculated.
    :return: the rel area in hectares of a square pixel of side length
    `pixel_size` whose center is at latitude `center_lat`.
    """

    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = np.sqrt(1 - (b/a)**2)
    area_list = []
    for f in [center_lat+pixel_size/2, center_lat-pixel_size/2]:
        zm = 1 - e*np.sin(np.radians(f))
        zp = 1 + e*np.sin(np.radians(f))
        area_list.append(
            np.pi * b**2 * (
                np.log(zp/zm) / (2*e) +
                np.sin(math.radians(f)) / (zp*zm)))
    return (pixel_size / 360. * (area_list[0] - area_list[1])) * np.power(10.0,-4)



def get_geometry_grid(geodataframe, epsg, shape):
    """
    get_geometry_grid creates a defined grid of the input territory extension.

    :param geodataframe: the territory vector file ina gdf format.
    :param epsg: the defined epsg of vector georeferenced data.
    :param shape: tuple describing the shape of the desired rasters within each grid tile.
    :return: the grid as a geodataframe
    """
    
    # Get the bounds of the territory.
    xmin, ymin, xmax, ymax = geodataframe.total_bounds

    # # Get ranges.
    # Deltax = np.abs(xmax - xmin)
    # Deltay = np.abs(ymax - ymin)

    # # Explicit dimensions of a raster within each grid tile.
    # lx = shape[0]
    # ly = shape[1]

    # # Grid dimensions.
    # height = np.ceil(Deltax/lx)
    # width = np.ceil(Deltay/ly)

    height = 10
    width = 10

    cols = list(np.arange(xmin, xmax + width, width))
    rows = list(np.arange(ymin, ymax + height, height))

    # Create a list of all the grid tiles.
    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x,y), (x+width, y), (x+width, y+height), (x, y+height)]))
    
    # Transform the polygon list into a GeoDataFrame.
    grid = gpd.GeoDataFrame({'geometry':polygons}, crs=epsg)
    
    return grid

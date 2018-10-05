#!/usr/bin/env
"""
This is a library to convert names to longitude and latitude, and vice versa
"""


def lon_lat_to_name(lon, lat):
    """
    converts lon (float) and lat (float) to 'lon_lat'
    """
    return "{0:f}_{1:f}".format(lon, lat)


def name_to_lon_lat(name):
    """
    Inverse of lon_lat_to_name
    """
    return float(name.split('_')[0]), float(name.split('_')[1])

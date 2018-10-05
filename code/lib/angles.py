#!/usr/bin/env
"""
This library deals with angles: unit change, angle distance, ...
"""
import numpy as np


def ang_dist(lon1, lat1, lon2, lat2):
    """
    Angular distance between 2 points, all arguments in rad.
    """
    ang_dist = np.sin(lat1)*np.sin(lat2)
    ang_dist += np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
    ang_dist = np.arccos(ang_dist)
    return ang_dist


def rad_to_deg(angle):
    """
    Conversion from radian to degree
    """
    return angle*180./np.pi


def deg_to_rad(angle):
    """
    Conversion from radian to degree
    """
    return angle*np.pi/180.


def convert_distance_to_physical(angular_distance_deg, units):
    """
    Radius of the Earth is 6371 km
    Adding a calibration factor from google map
    (https://www.google.com/maps/dir/Lexington+Ave+%26+E+61st+St,+New+York,
    +NY+10065/Park+Ave+%26+E+73rd+St,+New+York,+NY+10021/@40.7676217,
    -73.9701548,16z/data=!3m1!4b1!4m29!4m28!1m20!1m1!1s0x89c258ef6f253e81:
    0xc63aaaefe619a028!2m2!1d-73.9672732!2d40.763483!3m4!1m2!1d-73.9670875!
    2d40.7641824!3s0x89c258ef0e376ec5:0x684920ca0dae693c!3m4!1m2!1d-73.9670561
    !2d40.7658053!3s0x89c258eedd47f62f:0xbc3a1f4edbac2d31!3m4!1m2!1d
    -73.9648687!2d40.7674145!3s0x89c258ec0082592f:0x2c3535f29e0f6140!1m5!1m1!
    1s0x89c258eb33cd4015:0x777eea69b117a3c3!2m2!1d-73.9632455!2d40.7717443!3e2)
    This factor is 1.5567 km / 1.4000 km = 1.1119
    There is 0.621371 mile in a km
    """
    if units not in ['km', 'miles']:
        raise ValueError('Units must me "km" or "miles"')

    angular_distance_rad = deg_to_rad(angular_distance_deg)
    if units == 'km':
        angular_distance = angular_distance_rad * (6371. / 1.1119)
    else:
        angular_distance = angular_distance_rad * (6371. / 1.1119) * 0.621371
    return angular_distance


def convert_distance_to_degree(distance, units):
    """
    Radius of the Earth is 6371 km
    Adding a calibration factor from google map (see
    convert_distance_to_physical)
    This factor is 1.5567 km / 1.4000 km = 1.1119
    There is 0.621371 mile in a km
    """
    if units not in ['km', 'miles']:
        raise ValueError('Units must me "km" or "miles"')

    if units == 'km':
        distance_rad = distance / (6371. / 1.1119)
    else:
        distance_rad = distance / ((6371. / 1.1119) * 0.621371)
    distance_deg = rad_to_deg(distance_rad)
    return distance_deg

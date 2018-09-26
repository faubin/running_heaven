import numpy as np


def ang_dist(lon1, lat1, lon2, lat2):
    """
    Angular distance between 2 points, all arguments in rad.
    """
    ang_dist = np.sin(lat1)*np.sin(lat2)
    ang_dist += np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
    ang_dist = np.arccos(ang_dist)
    return ang_dist


def rad_to_deg(x):
    """
    Conversion from radian to degree
    """
    return x*180./np.pi


def deg_to_rad(x):
    """
    Conversion from radian to degree
    """
    return x*np.pi/180.

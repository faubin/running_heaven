import numpy as np
from running_heaven.code.lib import angles
from running_heaven.code.lib import names


def get_closest_point_to(lon0, lat0, intersection_list):
    """
    from a list of intersections (['lon_lat', ...]), returns the closest point
    to (lon, lat)
    Assumes all angles in degrees
    """
    locations = {'lon': [], 'lat': [], 'd': []}
    for name_ in intersection_list:
        lon, lat = names.name_to_lon_lat(name_)
        locations['lon'].append(lon)
        locations['lat'].append(lat)
        locations['d'].append(angles.ang_dist(angles.deg_to_rad(lon0),
                              angles.deg_to_rad(lat0), angles.deg_to_rad(lon),
                              angles.deg_to_rad(lat)))
    n = np.argmin(np.array(locations['d']))
    return intersection_list[n]


def get_angle(x1, y1, x2, y2):
    """
    returns the polar angle of point (x2, y2) with respect to (x1, y1) in rad
    """
    opp = y2 - y1
    adj = x2 - x1
    theta = np.arctan2(opp, adj)
    return theta

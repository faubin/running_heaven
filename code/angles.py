import pylab as pl


def ang_dist(lon1, lat1, lon2, lat2):
    """
    Angular distance between 2 points, all arguments in rad.
    """
    ang_dist = pl.sin(lat1)*pl.sin(lat2)
    ang_dist += pl.cos(lat1)*pl.cos(lat2)*pl.cos(lon2-lon1)
    ang_dist = pl.arccos(ang_dist)
    return ang_dist


def rad_to_deg(x):
    """
    Conversion from radian to degree
    """
    return x*180./pl.pi


def deg_to_rad(x):
    """
    Conversion from radian to degree
    """
    return x*pl.pi/180.

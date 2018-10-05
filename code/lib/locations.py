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


def select_random_point(segments, start_point, target_distance):
    """
    """
    # list of possible vertexes
    vertex_list = [v for i, v in enumerate(segments['vertex_start'].values) if segments['type'].iloc[i] == 'street']
    vertex_list = [v for i, v in enumerate(segments['vertex_end'].values) if segments['type'].iloc[i] == 'street']
    all_vertex = set(vertex_list)
    all_vertex = np.array([i for i in all_vertex])

    # getting lon and lat for all vertex
    all_vertex_lon_lat = {'lon': [], 'lat': []}
    for index, vertex in enumerate(all_vertex):
        lon, lat = names.name_to_lon_lat(vertex)
        all_vertex_lon_lat['lon'].append(lon)
        all_vertex_lon_lat['lat'].append(lat)
    all_vertex_lon_lat['lon'] = np.array(all_vertex_lon_lat['lon'])
    all_vertex_lon_lat['lat'] = np.array(all_vertex_lon_lat['lat'])

    # randmomly selectend point
    lon, lat = names.name_to_lon_lat(start_point)
    ang_dist_to_pt = angles.ang_dist(lon, lat,
                                     all_vertex_lon_lat['lon'],
                                     all_vertex_lon_lat['lat'])
    dist_to_pt = angles.convert_distance_to_physical(ang_dist_to_pt, 'km')
    valid = np.logical_and(dist_to_pt > 0., dist_to_pt < target_distance)

    index = int(np.floor(np.random.rand(1) * valid.sum())[0])
    end_point = all_vertex[valid][index]

    return end_point


def select_random_point_pairs(segments, target_distance, n_pairs=1):
    """
    """
    # list of possible vertexes
    vertex_list = [v for i, v in enumerate(segments['vertex_start'].values) if segments['type'].iloc[i] == 'street']
    vertex_list = [v for i, v in enumerate(segments['vertex_end'].values) if segments['type'].iloc[i] == 'street']
    all_vertex = set(vertex_list)
    all_vertex = np.array([i for i in all_vertex])

    # select n_pairs random indexes
    start_indexes = np.floor(np.random.rand(n_pairs) * len(all_vertex))
    start_indexes = start_indexes.astype(int)

    # find the points corresponding to the indices
    start_points = all_vertex[start_indexes]

    # getting lon and lat for all vertex
    all_vertex_lon_lat = {'lon': [], 'lat': []}
    for index, vertex in enumerate(all_vertex):
        lon, lat = names.name_to_lon_lat(vertex)
        all_vertex_lon_lat['lon'].append(lon)
        all_vertex_lon_lat['lat'].append(lat)
    all_vertex_lon_lat['lon'] = np.array(all_vertex_lon_lat['lon'])
    all_vertex_lon_lat['lat'] = np.array(all_vertex_lon_lat['lat'])

    end_points = []
    for i in range(len(start_points)):
        lon, lat = names.name_to_lon_lat(start_points[i])
        ang_dist_to_pt = angles.ang_dist(lon, lat,
                                         all_vertex_lon_lat['lon'],
                                         all_vertex_lon_lat['lat'])
        dist_to_pt = angles.convert_distance_to_physical(ang_dist_to_pt,
                                                         'km')
        valid = np.logical_and(dist_to_pt > 0., dist_to_pt < target_distance)

        index = int(np.floor(np.random.rand(1) * valid.sum())[0])
        end_points.append(all_vertex[valid][index])

    # points
    pts = []
    for i in range(n_pairs):
        pts.append((start_points[i], end_points[i]))

    return pts


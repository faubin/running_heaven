#!/usr/bin/env
"""
This class optimizes running routes based on tre density and parks given
distance constraints
"""
from collections import defaultdict
import copy
import heapq
import itertools
import os
import geopandas as gpd
import numpy as np
import pandas as pd
from running_heaven.code.lib import angles
from running_heaven.code.lib import core
from running_heaven.code.lib import data_handler
from running_heaven.code.lib import locations
from running_heaven.code.lib import map_plotter
from running_heaven.code.lib import names
try:
    import matplotlib.pyplot as pl
except ImportError:
    print('Warning: matplolib failed to run, run the app with show=False.')
try:
    import pulp
except ImportError:
    print('Warning: pulp was not imported, but it is not needed by default')


class RunRouteOptimizer(core.HeavenCore):
    """
    This class optimizes running routes based on tre density and parks given
    distance constraints
    """
    def __init__(self, num_points_per_segment=2, show=True):
        core.HeavenCore.__init__(self)
        self.data_hand = data_handler.DataHandler()
        self.num_points_per_segment = num_points_per_segment
        self.show = show
        return


    def define_variables(self, map_components):
        """
        Defines pulp variable for the integer programming

        Input:
            map_components is a pd.DataFrame of the raw data
        Outputs:
            segment_in_route is a list of pulp.LpVariable, 1 in route, 0 not
                in route
            segment_names is a list of the names of the variables
        """
        segment_in_route = []
        all_indices = range(len(map_components.index))
        for i in all_indices:
            path_name = '{0:s}_to_{1:s}'.format(map_components['vertex_start'].iloc[i],
                                                map_components['vertex_end'].iloc[i])
            path_name = path_name.replace('-', 'm')
            segment_in_route.append(pulp.LpVariable(path_name, 0, 1,
                                                    pulp.LpInteger))
        for i in all_indices:
            path_name = '{1:s}_to_{0:s}'.format(map_components['vertex_start'].iloc[i],
                                                map_components['vertex_end'].iloc[i])
            path_name = path_name.replace('-', 'm')
            segment_in_route.append(pulp.LpVariable(path_name, 0, 1,
                                                    pulp.LpInteger))
        return segment_in_route

    def integer_programming(self, map_components, units, route_end_labels,
                            target_distance):
        """
        Optimizes the route using integer programming

        Inputs:
            map_components is the raw data, a pd.DataFrame
            units is either 'km' or 'miles'
            route_end_labels is a tuple of the start and end labels
                'longitude_latitude'
            start_label is the 'longitude_latitude' label of the start point
            end_label is the 'longitude_latitude' label of the end point
            tartget distance is a float representing the target distance

        Outputs:
            segment_index, is a list of the indexes of map_components that are
                part of the route
            route_length is a float representing the total distance of the
                route in units
        """
        # add the tree and intersection cost
        cost_trees = map_components['tree_density_weight'].values
        cost_intersections = np.zeros(len(cost_trees))
        cost_intersections[map_components['type'].values == 'street'] = 1.
        costs = cost_trees + cost_intersections
        # add costs at the end of itself
        costs = np.append(costs, costs)

        # physical distances
        distance_deg = map_components['distance'].values
        distances = angles.convert_distance_to_physical(distance_deg, units)
        # add costs at the end of itself
        distances = np.append(distances, distances)

        # segment_in_route, is 1 for a selected path, 0 otherwise, need to
        # define both ways
        segment_in_route = self.define_variables(map_components)

        # Create the 'prob' variable to contain the problem data
        prob = pulp.LpProblem("Minimizing cost", pulp.LpMinimize)

        # The objective function is to minimize the cost function
        prob += (costs * segment_in_route).sum(), "Total cost"

        # closed loop constraint
        map_components.index = map_components.index.astype(int)
        all_vertex = set(list(map_components['vertex_start']) +
                         list(map_components['vertex_end']))
        vertex_starts = map_components['vertex_start'].values
        vertex_ends = map_components['vertex_end'].values
        for node in all_vertex:
            constraint = 0
            for i in map_components.index[(vertex_starts == node)]:
                constraint += segment_in_route[i]
            for i in map_components.index[(vertex_starts == node)]:
                constraint -= segment_in_route[i+len(map_components.index)]
            for i in map_components.index[(vertex_ends == node)]:
                constraint -= segment_in_route[i]
            for i in map_components.index[(vertex_ends == node)]:
                constraint += segment_in_route[i+len(map_components.index)]

            if node == route_end_labels[0]:
                prob += constraint == 1, 'Node {0:s}'.format(node)
            elif node == route_end_labels[1]:
                prob += constraint == -1, 'Node {0:s}'.format(node)
            else:
                prob += constraint == 0, 'Node {0:s}'.format(node)

        # can only go one way
        len_ = int(len(map_components.index))
        for i in range(0, len_):
            prob += segment_in_route[i] + segment_in_route[i+len_] <= 1,\
                    "one_way" + str(i)

        # constraint on distance
        prob += (segment_in_route * distances).sum() >= target_distance-0.2,\
                'distance1'
        prob += (segment_in_route * distances).sum() <= target_distance+0.2,\
                'distance2'

        # The problem is solved using PuLP's choice of Solver
        prob.solve()

        # The status of the solution is printed to the screen
        print("Status:", pulp.LpStatus[prob.status])

        # Each of the variables is printed with it's resolved optimum value
        segment_index = []
        for variable in prob.variables():
            if variable.varValue == 1.:
                vertex_name = variable.name.split('_to_')
                vertex_lon = vertex_name[0].replace('m', '-')
                vertex_lat = vertex_name[1].replace('m', '-')

                # normal order
                segment_valid = np.logical_and(vertex_starts == vertex_lon,
                                               vertex_ends == vertex_lat)
                index_selected = np.where(segment_valid)[0]
                if len(index_selected) == 1:
                    segment_index.append(index_selected[0])
                # reverse order
                segment_valid = np.logical_and(vertex_ends == vertex_lon,
                                               vertex_starts == vertex_lat)
                index_selected = np.where(segment_valid)[0]
                if len(index_selected) == 1:
                    segment_index.append(index_selected[0])

        # The optimised objective function value is printed to the screen
        # print("Total Cost = ", pulp.value(prob.objective))

        route_length_deg = map_components['distance'].iloc[segment_index].sum()
        route_length = angles.convert_distance_to_physical(route_length_deg,
                                                           units)
        return segment_index, route_length

    def update_costs(self, object_, target_d, d_done, current_point,
                     end_point, weight):
        """
        updates the costs in the defaultdict

        object_ is g in the dijkstra function
        target_d is the length of the intended run
        d_done is the dictionary of lengths so far for the routes

        cost function: ((d_left - d0) / d0)^2 * cos(theta_between_pt_and end)
                       ( 1 - (d_left - d0) / d0)^2 * tree_density_normalized

        Right now, sets all to 0
        """
        lon_end, lat_end = names.name_to_lon_lat(end_point)
        lon_current, lat_current = names.name_to_lon_lat(current_point)
        theta_end = locations.get_angle(lon_current, lat_current, lon_end,
                                        lat_end)

        for key_ in [current_point]:
            for i in range(len(object_[key_])):
                temp = list(object_[key_][i])
                lon, lat = names.name_to_lon_lat(object_[key_][i][1])

                # cost function
                dist_ran = temp[2]['distance'] + d_done[current_point]
                theta_pt = locations.get_angle(lon_current, lat_current, lon,
                                               lat)
                theta_diff = theta_end - theta_pt
                d_direct = angles.ang_dist(lon_current, lat_current, lon,
                                           lat)
                dist_left = target_d - dist_ran

                # cost function increases as run ends and is directed towards
                # the end point, has lower cost towards the end point
                # r_factor should be between 0 (start, dist_ran=0) and
                # 1 (end, dist_ran=target_d) if dist_ran < target_d:
                dist_frac = 0.5
                if dist_ran < dist_frac * target_d:
                    r_factor = (dist_ran / (dist_frac * target_d/2.))
                else:
                    r_factor = 1.
                # no cost as long as we have not reached the proper length
                if dist_left > d_direct:
                    cost_dist = 0.
                else:
                    cost_dist = r_factor**2
                    cost_dist *= ((1. + np.cos(np.pi+theta_diff))/2.)**2

                # spiral term when far
                cost_dist2 = (1. - r_factor)**2
                # cost_dist2 *= ((1. + np.cos(2.*theta_diff))/2.)**2
                cost_dist2 *= ((1. + np.cos(theta_diff))/2.)**2

                # tree weight
                cost_tree = (1. - r_factor)**2
                cost_tree *= (temp[2]['tree_density_weight'])**2
                # less cost for routes towards parks
                cost_park = (1. - r_factor)**2 * temp[2]['park_weight']**2

                # cost_intersection = (1. - r_factor)**2 * temp[2]['intersection']**2
                cost_intersection = temp[2]['intersection']**2

                cost_terms = [weight[0]*cost_dist,
                              weight[1]*cost_dist2,
                              weight[2]*cost_tree,
                              weight[3]*cost_park,
                              weight[4]*cost_intersection]

                temp[0] = copy.deepcopy(np.sum(cost_terms))
                object_[key_][i] = temp
        return object_

    def dijkstra(self, edges, start_label, final_label, target_distance,
                 weight):
        """
        edges is a list of ('pt1_label', 'pt2_label', cost), cost is typically
        distance
        start_label is the start point
        end_label is the final point
        """
        vertex_connect = defaultdict(list)
        for label1, label2, cost_segment, info in edges:
            vertex_connect[label1].append((cost_segment, label2, info))

        vertexes = [(0, start_label, (), 0)]
        seen = set()
        mins_cost = {start_label: 0}
        calc_dists = {start_label: 0}
        vertex_connect = self.update_costs(vertex_connect,
                                           target_distance,
                                           calc_dists,
                                           start_label,
                                           final_label,
                                           weight)

        while vertexes:
            # get a new vertex to visit
            (cost, label_at, path, dist) = heapq.heappop(vertexes)

            # do not revisit vertex
            if label_at not in seen:
                seen.add(label_at)
                path = (label_at, path)

                # if arrived to last vertex, done
                if label_at == final_label:
                    return (cost, path, dist)

                vertex_connect = self.update_costs(vertex_connect,
                                                   target_distance,
                                                   calc_dists,
                                                   label_at,
                                                   final_label,
                                                   weight)
                for cost_, label_next, info_ in vertex_connect.get(label_at,
                                                                   ()):

                    # do not repeat visited vertex
                    if label_next in seen:
                        continue

                    prev_cost = mins_cost.get(label_next, None)
                    new_cost = cost + cost_
                    new_d = dist + info_['distance']
                    if prev_cost is None or new_cost < prev_cost:
                        mins_cost[label_next] = new_cost
                        calc_dists[label_next] = new_d
                        heapq.heappush(vertexes, (new_cost,
                                                  label_next,
                                                  path,
                                                  new_d))

        # return float("inf")
        return (cost, path, dist)

    # def define_park_weight(self, df1, df2, target_d):
    def define_park_weight(self, segments, target_d):
        """
        """
        weight = (segments['min_dist_to_park'] / target_d)**0.5
        weight.loc[weight > 1.] = 1.
        return weight

    def feature_distributions(self, tree_density_norm, park_weight):
        """
        Plots the distributions of features
        """
        pl.figure(1)
        pl.subplot(311)
        pl.hist(tree_density_norm)
        pl.ylabel('Count')
        pl.xlabel('# trees per segment / max(# trees per segment)')
        pl.xlim([0., 1.])
        pl.subplot(313)
        pl.hist(park_weight)
        pl.ylabel('Count')
        pl.xlabel('Distance to nearest park / run length target')
        pl.xlim([0., 1.])
        pl.savefig('distributions.png')
        pl.close(1)
        return

    def find_vertex_index(self, segments, pt1, pt2):
        """
        """
        ind = segments["vertex_start"] == pt1
        ind &= segments['vertex_end'] == pt2
        ind = np.where(ind)[0]
        return ind

    def get_indices_from_path(self, path, start_pt, segments):
        """
        """
        dist = path[2]
        path = path[1]
        prev_pt = None
        path_indices = []
        while not path[0] == start_pt:
            # nothing to do for first vertex
            if prev_pt is not None:
                ind = self.find_vertex_index(segments, prev_pt, path[0])
                if len(ind) > 1:
                    print('problem with selecting path', ind)
                path_indices.append(ind[0])

            # update values for next loop
            prev_pt = copy.deepcopy(path[0])
            path = path[1]
        ind = self.find_vertex_index(segments, prev_pt, path[0])
        if len(ind) > 1:
            print('problem with selecting path', ind)
        path_indices.append(ind[0])
        return path_indices, dist

    def get_route(self, segments, path_indices, start_point, end_point):
        """
        """
        vertexes = []
        geometries = []
        for path_index in path_indices:
            name = '{0:s}_to_{1:s}'.format(segments['vertex_start'].iloc[path_index],
                                           segments['vertex_end'].iloc[path_index])
            vertexes.append(name)
            geometries.append(segments['geometry'].iloc[path_index])
            name = '{1:s}_to_{0:s}'.format(segments['vertex_start'].iloc[path_index],
                                           segments['vertex_end'].iloc[path_index])
            vertexes.append(name)
            geometries.append(segments['geometry'].iloc[path_index])

        node = start_point
        path = []
        done = False
        while not done:
            prev_path_length = len(path)
            for vertex_index, vertex in enumerate(vertexes):
                if vertex.split('_to_')[0] == node:
                    path.append(vertex.split('_to_')[0].split('_')[::-1])
                    node = vertex.split('_to_')[1]

                    # adds more resulution
                    for n_pt in range(1, self.num_points_per_segment):
                        n_pts = len(geometries[vertex_index].xy[0])
                        index_ = int(n_pt * n_pts/ self.num_points_per_segment)
                        path.append([geometries[vertex_index].xy[1][index_],
                                     geometries[vertex_index].xy[0][index_]])

                    # removes values when used
                    if vertex_index % 2 == 1:
                        vertexes.pop(vertex_index)
                        vertexes.pop(vertex_index-1)
                        geometries.pop(vertex_index)
                        geometries.pop(vertex_index-1)
                    else:
                        vertexes.pop(vertex_index+1)
                        vertexes.pop(vertex_index)
                        geometries.pop(vertex_index+1)
                        geometries.pop(vertex_index)
                    break
            if node == end_point:
                done = True
            if len(path) == prev_path_length:
                return None
        path.append(end_point.split('_')[::-1])
        return path

    def run(self, route_ends, target_dist, units='km',
            algorithm_type='dijkstra', cost_weights=None):
        """
        """
        segments = self.data_hand.load_processed_data()

        # list of all possible intersections
        intersection_names = list(segments['vertex_start'].values)
        intersection_names += list(segments['vertex_end'].values)
        intersection_names = list(set(intersection_names))

        # get closest intersection to provided points
        start_point_lon, start_point_lat = names.name_to_lon_lat(route_ends[0])
        end_point_lon, end_point_lat = names.name_to_lon_lat(route_ends[1])
        start_point = locations.get_closest_point_to(start_point_lon,
                                                     start_point_lat,
                                                     intersection_names)
        end_point = locations.get_closest_point_to(end_point_lon,
                                                   end_point_lat,
                                                   intersection_names)
        route_ends = [start_point, end_point]
        print("Optimizing route from {0:s} to {1:s}".format(start_point,
                                                            end_point))

        # load data for plotting
        dfs = {}
        processed_path = os.path.join(self.running_heaven_path, 'data',
                                      'processed')
        for key_ in ['park', 'street', 'sidewalk']:
            dfs[key_] = gpd.read_file(os.path.join(processed_path,
                                                   '{0:s}.geojson'.format(key_)))
        dfs['tree'] = pd.read_csv(os.path.join(processed_path, 'tree.csv'))

        # updating dataframe
        segments['tree_density_weight'] = 1. - segments['tree_density']
        target_dist_deg = angles.convert_distance_to_degree(target_dist, units)
        park_weight = self.define_park_weight(segments,  # dfs['park'],
                                              target_dist_deg)
        segments['park_weight'] = park_weight

        # distribution of features for debugging
        # self.feature_distributions(tree_density_norm, park_weight)

        # linear programming solution
        if algorithm_type == 'integer_programming':
            path_indices, d_path = self.integer_programming(segments,
                                                            units,
                                                            (start_point,
                                                             end_point),
                                                            target_dist)
        elif algorithm_type == 'dijkstra':
            # problem information for Dijkstra's algorithm
            # add the segments in the reverse direction as well
            edges = []
            segments_copy = copy.deepcopy(segments)
            vertex_start = copy.deepcopy(segments_copy['vertex_start'])
            segments_copy['vertex_start'] = copy.deepcopy(segments_copy['vertex_end'])
            segments_copy['vertex_end'] = copy.deepcopy(vertex_start)
            segments = segments.append(segments_copy)
            segments.reset_index(drop=True, inplace=True)

            for index in segments.index.astype(int):
                # distance - shortest path
                edges.append((segments['vertex_start'].iloc[index],  # starting point
                              segments['vertex_end'].iloc[index],  # end point
                              0.,  # cost, updated automatically
                              {'distance': segments['distance'].iloc[index],
                               'tree_density_weight': segments['tree_density_weight'].iloc[index],
                               'park_weight': segments['park_weight'].iloc[index],
                               'intersection': int(segments['type'].iloc[index] == 'street'),
                              }
                             ))

            # try different cost term weights
            choices = [0.1, 10.]
            weights = [np.array(p) for p in itertools.product(choices,
                                                              repeat=5)]
            # set fixed values if provides
            # distance, spiral, tree, park, intersection
            if cost_weights is not None:
                fixed = ~np.isnan(cost_weights)
                for weight in weights:
                    weight[fixed] = np.array(cost_weights)[fixed]
                weights = [list(i) for i in weights]

                # remove duplicates
                for i in range(len(weights)-1, -1, -1):
                    if weights.count(weights[i]) > 1:
                        weights.pop(i)

            # iterate on the different weights
            path_indices_list = []
            d_path_list = []
            cost_list = []
            for weight in weights:
                opt_path = self.dijkstra(edges, start_point, end_point,
                                         target_dist_deg, weight)

                if opt_path[0] == 0.:
                    print('Warning: impossible route')
                    return None, None, None

                # get indices from path
                path_indices, d_path = self.get_indices_from_path(opt_path,
                                                                  start_point,
                                                                  segments)
                d_path = angles.convert_distance_to_physical(d_path, units)
                path_indices_list.append(path_indices)
                d_path_list.append(d_path)
                cost_list.append(opt_path[0])
                # print(weight, cost_list[-1], d_path)

            distance_difference = abs(np.array(d_path_list) - target_dist)
            index_closest_distance = np.argmin(distance_difference)
            d_path = d_path_list[index_closest_distance]
            path_indices = path_indices_list[index_closest_distance]
        else:
            exit('Analysis types 1 and 2 defined so far.')

        # plotting the data and route
        if self.show:
            map_plotter.plot_route(dfs, route_ends, segments, path_indices)

        # resulting distance
        print('Total distance is : {0:f} {1:s}'.format(d_path, units))
        print('Taget distance was: {0:f} {1:s}'.format(target_dist, units))

        if self.show:
            pl.show()

        route_lon_lat = self.get_route(segments, path_indices, start_point,
                                       end_point)

        return d_path, route_lon_lat, segments.iloc[path_indices]

if __name__ == "__main__":
    # ROUTE_ENDS = ('-73.967_40.763', '-73.963_40.772')  # SE NE of CP
    # ROUTE_ENDS = ('-73.963_40.772', '-73.967_40.763')
    # ROUTE_ENDS = ('-73.994_40.740', '-73.995_40.749')

    # central park
    ROUTE_ENDS = ('-73.967_40.763', '-73.979_40.777')  # SE to NW of CP
    # ROUTE_ENDS = ('-73.967_40.763', '-73.967_40.764')  # SE to SE of CP
    # loops in central park
    # ROUTE_ENDS = ('-73.976_40.766', '-73.980_40.769')  # loop in CP

    # south Mahattan
    # ROUTE_ENDS = ('-73.988_40.729', '-73.996_40.722')  #
    # ROUTE_ENDS = ('-73.974_40.726', '-73.985_40.7112')  #

    # one point is sidewalk in Brooklyn
    # ROUTE_ENDS=('40.776112_-73.979746', '40.778238_-73.971427')

    COST_WEIGHTS = [np.nan, np.nan, np.nan, np.nan, np.nan]
    # cost_weights = [np.nan, np.nan, 0., np.nan, np.nan]
    # cost_weights = [np.nan, np.nan, np.nan, 0., np.nan]
    # cost_weights = [np.nan, np.nan, np.nan, np.nan, 0.]
    # cost_weights = [np.nan, np.nan, 0., 0., np.nan]
    # cost_weights = [np.nan, np.nan, 0., np.nan, 0.]
    # cost_weights = [np.nan, np.nan, np.nan, 0., 0.]
    # cost_weights = [np.nan, np.nan, 0., 0., 0.]

    TARGET_DISTANCE = 3.

    UNITS = 'km'
    # UNITS = 'miles'

    ALGORITHM_TYPE = 'dijkstra'
    # ALGORITHM_TYPE = 'integer_programming'

    APP = RunRouteOptimizer()
    APP.run(ROUTE_ENDS, TARGET_DISTANCE, UNITS, ALGORITHM_TYPE,
            cost_weights=COST_WEIGHTS)

from collections import defaultdict
import copy
import geopandas as gpd
import numpy as np
import heapq
import itertools
import os
import pandas as pd
import pdb
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
    """
    def __init__(self, show=True, borough='M'):
        """
        """
        core.HeavenCore.__init__(self)
        self.data_hand = data_handler.DataHandler()
        self.show = show
        return

    def int_prog(self, df, units, start_label, end_label, target_distance,
                 park):
        """
        """
        costs = df['tree_density_weight'].values
        cost_intersections = np.zeros(len(costs))
        cost_intersections[df['type'].values == 'street'] = 1.
        costs += cost_intersections
        costs = np.append(costs, costs)
        distances = angles.convert_distance_to_physical(df['distance'].values,
                                                        units)
        distances = np.append(distances, distances)

        # x is 1 for a selected path, 0 otherwise, need to define both ways
        names = []
        x = []
        all_indices = range(len(df.index))
        for i in all_indices:
            path_name = '{0:s}_to_{1:s}'.format(df['vertex_start'].iloc[i],
                                                df['vertex_end'].iloc[i])
            path_name = path_name.replace('-', 'm')
            names.append(path_name)
            x.append(pulp.LpVariable(path_name, 0, 1, pulp.LpInteger))
        for i in all_indices:
            path_name = '{1:s}_to_{0:s}'.format(df['vertex_start'].iloc[i],
                                                df['vertex_end'].iloc[i])
            path_name = path_name.replace('-', 'm')
            names.append(path_name)
            x.append(pulp.LpVariable(path_name, 0, 1, pulp.LpInteger))

        # Create the 'prob' variable to contain the problem data
        prob = pulp.LpProblem("Minimizing cost", pulp.LpMinimize)

        # The objective function is to minimize the cost function
        prob += (costs * x).sum(), "Total cost"

        # closed loop constraint
        df.index = df.index.astype(int)
        all_vertex = set(list(df['vertex_start']) + list(df['vertex_end']))
        for node in all_vertex:
            constraint = 0
            for i in df.index[(df['vertex_start'] == node)]:
                constraint += x[i]
            for i in df.index[(df['vertex_start'] == node)]:
                constraint -= x[i+len(df.index)]
            for i in df.index[(df['vertex_end'] == node)]:
                constraint -= x[i]
            for i in df.index[(df['vertex_end'] == node)]:
                constraint += x[i+len(df.index)]

            if node == start_label:
                prob += constraint == 1, 'Node {0:s}'.format(node)
            elif node == end_label:
                prob += constraint == -1, 'Node {0:s}'.format(node)
            else:
                prob += constraint == 0, 'Node {0:s}'.format(node)

        # can only go one way
        len_ = int(len(df.index))
        for i in range(0, len_):
            prob += x[i] + x[i+len_] <= 1, "one_way" + str(i)

        # constraint on distance
        # prob += (x * distances).sum() >= 0.9 * target_distance, 'distance1'
        # prob += (x * distances).sum() <= 1.1 * target_distance, 'distance2'
        prob += (x * distances).sum() >= target_distance-0.2, 'distance1'
        prob += (x * distances).sum() <= target_distance+0.2, 'distance2'

        # The problem is solved using PuLP's choice of Solver
        prob.solve()

        # The status of the solution is printed to the screen
        print("Status:", pulp.LpStatus[prob.status])

        # Each of the variables is printed with it's resolved optimum value
        n = 0
        inde = []
        vert = []
        for v in prob.variables():
            n += 1
            if v.varValue == 1.:
                print(v.name, "=", v.varValue)
                vert.append(v.name)
                vert_names = vert[-1].split('_to_')
                for i in range(len(vert_names)):
                    vert_names[i] = vert_names[i].replace('m', '-')

                xxx = np.where(np.logical_and(df['vertex_start'].values == vert_names[0], df['vertex_end'].values == vert_names[1]))[0]
                if len(xxx) == 1:
                    inde.append(xxx[0])
                xxx = np.where(np.logical_and(df['vertex_end'].values == vert_names[0], df['vertex_start'].values == vert_names[1]))[0]
                if len(xxx) == 1:
                    inde.append(xxx[0])

        # The optimised objective function value is printed to the screen
        print("Total Cost = ", pulp.value(prob.objective))

        length = angles.convert_distance_to_physical(df['distance'].iloc[inde].sum(), units)
        return inde, length

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

                # angular distance to end
                ang_dist = angles.ang_dist(lon_end, lat_end, lon, lat)

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
        g = defaultdict(list)
        for label1, label2, cost_segment, info in edges:
            g[label1].append((cost_segment, label2, info))

        q = [(0, start_label, (), 0)]
        seen = set()
        mins_cost = {start_label: 0}
        calc_dists = {start_label: 0}
        g = self.update_costs(g, target_distance, calc_dists, start_label,
                              final_label, weight)

        while q:
            # get a new vertex to visit
            temp = heapq.heappop(q)
            (cost, v1, path, dist) = temp

            # do not revisit vertex
            if v1 not in seen:
                seen.add(v1)
                path = (v1, path)

                # if arrived to last vertex, done
                if v1 == final_label:
                    return (cost, path, dist)

                g = self.update_costs(g, target_distance, calc_dists, v1,
                                      final_label, weight)
                for c, v2, info_ in g.get(v1, ()):

                    # do not repeat visited vertex
                    if v2 in seen:
                        continue

                    prev_cost = mins_cost.get(v2, None)
                    prev_dist = calc_dists.get(v2, None)
                    new_cost = cost + c
                    new_d = dist + info_['distance']
                    if prev_cost is None or new_cost < prev_cost:
                        mins_cost[v2] = new_cost
                        calc_dists[v2] = new_d
                        heapq.heappush(q, (new_cost, v2, path, new_d))

        # return float("inf")
        return (cost, path, dist)

    # def define_park_weight(self, df1, df2, target_d):
    def define_park_weight(self, df, target_d):
        """
        """
        ret = []
        weight = (df['min_dist_to_park'] / target_d)**0.5  # **2
        # weight = (df['min_dist_to_park'] / target_d)**1.0  # **2
        # weight = (df['min_dist_to_park'] / target_d)**2.0  # **2
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

    def find_vertex_index(self, df, pt1, pt2):
        """
        """
        ind = df["vertex_start"] == pt1
        ind &= df['vertex_end'] == pt2
        ind = np.where(ind)[0]
        return ind

    def get_indices_from_path(self, path, start_pt, df):
        """
        """
        dist = path[2]
        path = path[1]
        prev_pt = None
        path_indices = []
        while not path[0] == start_pt:
            # nothing to do for first vertex
            if prev_pt is not None:
                ind = self.find_vertex_index(df, prev_pt, path[0])
                if len(ind) > 1:
                    print('problem with selecting path', ind)
                path_indices.append(ind[0])

            # update values for next loop
            prev_pt = copy.deepcopy(path[0])
            path = path[1]
        ind = self.find_vertex_index(df, prev_pt, path[0])
        if len(ind) > 1:
            print('problem with selecting path', ind)
        path_indices.append(ind[0])
        return path_indices, dist

    def get_route(self, df, path_indices, start_point, end_point):
        """
        """
        vertexes = []
        for i in range(len(path_indices)):
            name = '{0:s}_to_{1:s}'.format(df['vertex_start'].iloc[path_indices[i]],
                                           df['vertex_end'].iloc[path_indices[i]])
            vertexes.append(name)
            name = '{1:s}_to_{0:s}'.format(df['vertex_start'].iloc[path_indices[i]],
                                           df['vertex_end'].iloc[path_indices[i]])
            vertexes.append(name)

        node = start_point
        path = []
        done = False
        while not done:
            prev_path_length = len(path)
            for i in range(len(vertexes)):
                if vertexes[i].split('_to_')[0] == node:
                    path.append(vertexes[i].split('_to_')[0].split('_')[::-1])
                    node = vertexes[i].split('_to_')[1]
                    mod = i % 2
                    if i % 2 == 1:
                        vertexes.pop(i)
                        vertexes.pop(i-1)
                    else:
                        vertexes.pop(i+1)
                        vertexes.pop(i)
                    break
            if node == end_point:
                done = True
            if len(path) == prev_path_length:
                return None
        path.append(end_point.split('_')[::-1])
        return path

    def run(self, pts, target_dist, units='km', type_=1, cost_weights=None):
        """
        """
        df_proc = self.data_hand.load_processed_data()

        # list of all possible intersections
        intersection_names = list(df_proc['vertex_start'].values)
        intersection_names += list(df_proc['vertex_end'].values)
        intersection_names = list(set(intersection_names))

        # get closest intersection to provided points
        start_point_lon, start_point_lat = names.name_to_lon_lat(pts[0])
        end_point_lon, end_point_lat = names.name_to_lon_lat(pts[1])
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
        df_proc['tree_density_weight'] = 1. - df_proc['tree_density']
        target_dist_deg = angles.convert_distance_to_degree(target_dist, units)
        park_weight = self.define_park_weight(df_proc,  # dfs['park'],
                                              target_dist_deg)
        df_proc['park_weight'] = park_weight

        # distribution of features for debugging
        # self.feature_distributions(tree_density_norm, park_weight)

        # linear programming solution
        if type_ == 2:
            path_indices, d_path = self.int_prog(df_proc, units, start_point,
                                                 end_point, target_dist,
                                                 dfs['park'])
        elif type_ == 1:
            # problem information for Dijkstra's algorithm
            edges = []
            new_df3 = copy.deepcopy(df_proc)
            st = copy.deepcopy(new_df3['vertex_start'])
            new_df3['vertex_start'] = copy.deepcopy(new_df3['vertex_end'])
            new_df3['vertex_end'] = copy.deepcopy(st)
            df_proc = df_proc.append(new_df3)
            for i in range(len(df_proc.index)):
                # distance - shortest path
                edges.append((df_proc['vertex_start'].iloc[i],  # starting point
                              df_proc['vertex_end'].iloc[i],  # end point
                              0.,  # cost, updated automatically
                              {'distance': df_proc['distance'].iloc[i],
                               'tree_density_weight': df_proc['tree_density_weight'].iloc[i],
                               'park_weight': df_proc['park_weight'].iloc[i],
                               'intersection': int(df_proc['type'].iloc[i] == 'street'),
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
                for i in range(len(weights)):
                    weights[i][fixed] = np.array(cost_weights)[fixed]
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
                                                                  df_proc)
                d_path = angles.convert_distance_to_physical(d_path, units)
                path_indices_list.append(path_indices)
                d_path_list.append(d_path)
                cost_list.append(opt_path[0])
                # print(weight, cost_list[-1], d_path)

            n = np.argmin(abs(np.array(d_path_list) - target_dist))
            # n = np.argmin(cost_list)
            # print(n, weights[n])
            d_path = d_path_list[n]
            path_indices = path_indices_list[n]
        else:
            exit('Analysis types 1 and 2 defined so far.')

        # plotting the data and route
        if self.show:
            map_plotter.plot_route(dfs, route_ends, df_proc, path_indices)

        # resulting distance
        print('Total distance is : {0:f} {1:s}'.format(d_path, units))
        print('Taget distance was: {0:f} {1:s}'.format(target_dist, units))

        if self.show:
            pl.show()

        route_lon_lat = self.get_route(df_proc, path_indices, start_point,
                                       end_point)

        return d_path, route_lon_lat, df_proc.iloc[path_indices]

if __name__ == "__main__":
    # pt1 = Lexington Ave & E 61st St, New York, NY 10065
    # pt2 = Park Ave & E 79th St, New York, NY 10075
    # pts = ('-73.967_40.763', '-73.963_40.772')  # SE NE of CP
    # pts = ('-73.963_40.772', '-73.967_40.763')
    # pts = ('-73.994_40.740', '-73.995_40.749')

    # central park
    pts = ('-73.967_40.763', '-73.979_40.777')  # SE to NW of CP
    # pts = ('-73.967_40.763', '-73.967_40.764')  # SE to SE of CP
    # loops in central park
    # pts = ('-73.976_40.766', '-73.980_40.769')  # loop in CP

    # south Mahattan
    # pts = ('-73.988_40.729', '-73.996_40.722')  #
    # pts = ('-73.974_40.726', '-73.985_40.7112')  #

    # one point is sidewalk in Brooklyn
    # pts=('40.776112_-73.979746', '40.778238_-73.971427')

    cost_weights = [np.nan, np.nan, np.nan, np.nan, np.nan]
    # cost_weights = [np.nan, np.nan, 0., np.nan, np.nan]
    # cost_weights = [np.nan, np.nan, np.nan, 0., np.nan]
    # cost_weights = [np.nan, np.nan, np.nan, np.nan, 0.]
    # cost_weights = [np.nan, np.nan, 0., 0., np.nan]
    # cost_weights = [np.nan, np.nan, 0., np.nan, 0.]
    # cost_weights = [np.nan, np.nan, np.nan, 0., 0.]
    # cost_weights = [np.nan, np.nan, 0., 0., 0.]

    target_dist = 3.

    units = 'km'
    # units = 'miles'

    type_ = 1  # Dijkstra's algorithm
    # type_ = 2  # intger programming, slow and has problems

    app = RunRouteOptimizer()
    d = app.run(pts, target_dist, units, type_, cost_weights=cost_weights)

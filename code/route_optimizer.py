from collections import defaultdict
import heapq
import copy
import pandas as pd
import geopandas as gpd
import pylab as pl
import names
import angles
import locations
import pdb


class RunRouteOptimizer():
    """
    """
    def __init__(self, show=True, borough='M'):
        """
        """
        self.show = show
        return

    def update_costs(self, object_, target_d, d_done, current_point,
                     end_point):
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
                new_dist = temp[2]['distance'] + d_done[current_point]
                theta_pt = locations.get_angle(lon_current, lat_current, lon,
                                               lat)
                theta_diff = theta_end - theta_pt
                d_direct = angles.ang_dist(lon_current, lat_current, lon,
                                           lat)
                dist_left = target_d - new_dist

                # cost function increases as run ends and is directed towards
                # the end point, has lower cost towards the end point
                # r_factor should be between 0 (start, new_dist=0) and
                # 1 (end, new_dist=target_d) if new_dist < target_d:
                if new_dist < target_d:
                    r_factor = dist_left / target_d
                else:
                    r_factor = 1.
                # no cost as long as we have not reached the proper length
                if dist_left > d_direct:
                    cost_dist = 0.
                else:
                    cost_dist = (1. - r_factor)**2
                    cost_dist *= ((1. + pl.cos(pl.pi+theta_diff))/2.)**2

                # spiral term when far
                cost_dist2 = (r_factor)**2
                # cost_dist2 *= ((1. + pl.cos(2.*theta_diff))/2.)**2
                cost_dist2 *= ((1. + pl.cos(theta_diff))/2.)**2

                # tree weight
                cost_tree = (r_factor)**2 * (temp[2]['tree_density_weight'])**2
                # less cost for routes towards parks
                cost_park = (r_factor)**2 * temp[2]['park_weight']**2
                cost_terms = [cost_dist, cost_dist2, cost_tree, cost_park]

                temp[0] = copy.deepcopy(pl.sum(cost_terms))
                object_[key_][i] = temp
        return object_

    def dijkstra(self, edges, start_label, final_label, target_distance):
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
                              final_label)

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
                                      final_label)
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

        return float("inf")

    # def define_park_weight(self, df1, df2, target_d):
    def define_park_weight(self, df, target_d):
        """
        """
        ret = []
        weight = (df['min_dist_to_park'] / target_d)**0.5  # **2
        #weight = (df['min_dist_to_park'] / target_d)**1.0  # **2
        #weight = (df['min_dist_to_park'] / target_d)**2.0  # **2
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

    def plot_route(self, dfs, intersection_names, start_point, end_point,
                   new_df2, path_indices):

        # plotting the map
        fig, ax2 = pl.subplots(figsize=(12, 9))
        colors = 'yrb'
        for n_key, key_ in enumerate(['park', 'street', 'sidewalk']):
            dfs[key_].plot(ax=ax2, color=colors[n_key])
        pl.plot(dfs['tree']['longitude'], dfs['tree']['latitude'], '.g',
                markersize=2)
        pl.xlabel('Longitude ($^o$)', fontsize=20)
        pl.ylabel('Latitude ($^o$)', fontsize=20)
        pl.xticks(fontsize=16)
        pl.yticks(fontsize=16)
        pl.title('NYC Map of Running Areas', fontsize=20)

        # plots starting and end point
        lon_start, lat_start = names.name_to_lon_lat(start_point)
        pl.plot(lon_start, lat_start, 'sc', markersize=12)
        lon_end, lat_end = names.name_to_lon_lat(end_point)
        pl.plot(lon_end, lat_end, 'oc', markersize=12)

        # black dot on all intersections
        # for i in intersection_names:
        #     lon, lat = names.name_to_lon_lat(i)
        #     pl.plot(lon, lat, '.k')

        # plotting the route
        new_gdf2 = gpd.GeoDataFrame(new_df2)
        new_gdf2_path = new_gdf2.iloc[pl.array(path_indices)]
        new_gdf2_path.plot(ax=ax2, color='k', linewidth=4)
        pl.savefig('path_run.png')
        pl.savefig('../app/flaskexample/static/path_run.png')
        return

    def find_vertex_index(self, df, pt1, pt2):
        """
        """
        ind = df["vertex_start"] == pt1
        ind &= df['vertex_end'] == pt2
        ind = pl.where(ind)[0]
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

    def convert_distance_to_physical(self, d, units):
        """
        Radius of the Earth is 6371 km
        Adding a calibration factor from google map
        (https://www.google.com/maps/dir/Lexington+Ave+%26+E+61st+St,+New+York,+NY+10065/Park+Ave+%26+E+73rd+St,+New+York,+NY+10021/@40.7676217,-73.9701548,16z/data=!3m1!4b1!4m29!4m28!1m20!1m1!1s0x89c258ef6f253e81:0xc63aaaefe619a028!2m2!1d-73.9672732!2d40.763483!3m4!1m2!1d-73.9670875!2d40.7641824!3s0x89c258ef0e376ec5:0x684920ca0dae693c!3m4!1m2!1d-73.9670561!2d40.7658053!3s0x89c258eedd47f62f:0xbc3a1f4edbac2d31!3m4!1m2!1d-73.9648687!2d40.7674145!3s0x89c258ec0082592f:0x2c3535f29e0f6140!1m5!1m1!1s0x89c258eb33cd4015:0x777eea69b117a3c3!2m2!1d-73.9632455!2d40.7717443!3e2)
        This factor is 1.5567 km / 1.4000 km = 1.1119
        There is 0.621371 mile in a km
        """
        if units not in ['km', 'miles']:
            raise ValueError('Units must me "km" or "miles"')

        d = angles.deg_to_rad(d)
        if units == 'km':
            d *= (6371. / 1.1119)
        else:
            d *= (6371. / 1.1119) * 0.621371
        return d

    def run(self, pt1, pt2, target_dist_deg, units='km'):
        """
        """
        new_df2 = gpd.read_file('processed/route_connections.geojson')
        new_df2.rename(index=str, columns={'vertex_sta': 'vertex_start',
                                           'tree_numbe': 'tree_number',
                                           'tree_densi': 'tree_density',
                                           'min_dist_t': 'min_dist_to_park'},
                       inplace=True)

        # list of all possible intersections
        intersection_names = list(new_df2['vertex_start'].values)
        intersection_names += list(new_df2['vertex_end'].values)
        intersection_names = list(set(intersection_names))

        # get closest intersection to provided points
        pt1_lon, pt1_lat = names.name_to_lon_lat(pt1)
        pt2_lon, pt2_lat = names.name_to_lon_lat(pt2)
        start_point = locations.get_closest_point_to(pt1_lon, pt1_lat,
                                                     intersection_names)
        end_point = locations.get_closest_point_to(pt2_lon, pt2_lat,
                                                   intersection_names)
        print("Optimizing route from {0:s} to {1:s}".format(pt1, pt2))

        # tree weight
        tree_density_weight = 1. -  new_df2['tree_density']

        # load data for plotting
        dfs = {}
        for key_ in ['park', 'street', 'sidewalk']:
            print(key_)
            dfs[key_] = gpd.read_file('processed/{0:s}.geojson'.format(key_))
        print('tree')
        dfs['tree'] = pd.read_csv('processed/tree.csv')
        print('tree_weights')
        park_weight = self.define_park_weight(new_df2,# dfs['park'],
                                              target_dist_deg)

        # distribution of features for debugging
        # self.feature_distributions(tree_density_norm, park_weight)

        print('optimization setup')
        # problem information for Dijkstra's algorithm
        edges = []
        for i in range(len(new_df2.index)):
            # distance - shortest path
            edges.append((new_df2['vertex_start'].iloc[i],  # starting point
                          new_df2['vertex_end'].iloc[i],  # end point
                          0.,  # cost, updated automatically
                          {'distance': new_df2['distance'].iloc[i],
                           'tree_density_weight': tree_density_weight[i],
                           'park_weight': park_weight[i],
                           }
                          ))

        print('optimization')
        # optimizing path
        opt_path = self.dijkstra(edges, start_point, end_point,
                                 target_dist_deg)
        print(opt_path)

        print('extracting path')
        # get indices from path
        path_indices, d_path = self.get_indices_from_path(opt_path, start_point,
                                                          new_df2)
        print(path_indices)

        print('plotting')
        # plotting the data and route
        self.plot_route(dfs, intersection_names, start_point, end_point,
                        new_df2, path_indices)

        # resulting distance
        print('Total distance is : {0:f} degrees'.format(d_path))
        print('Taget distance was: {0:f} degrees'.format(target_dist_deg))
        d_path_physical = self.convert_distance_to_physical(d_path, units)
        print('Total distance is : {0:f} {1:s}'.format(d_path_physical, units))
        d_path_target = self.convert_distance_to_physical(target_dist_deg,
                                                          units)
        print('Taget distance was: {0:f} {1:s}'.format(d_path_target, units))

        if self.show:
            pl.show()

        return d_path

if __name__ == "__main__":
    # pt1, pt2 = (-73.978, 40.778, -73.967, 40.767)

    # pt1 = Lexington Ave & E 61st St, New York, NY 10065
    # pt2 = Park Ave & E 79th St, New York, NY 10075
    pt1, pt2 = ('-73.967_40.763', '-73.963_40.772')
    # pt2, pt1 = ('-73.967_40.763', '-73.963_40.772')

    target_dist_deg = 0.011  # shortest distance east of Central Park
    target_dist_deg += 0.005
    target_dist_deg *= 2.

    units = 'km'
    # units = 'miles'

    app = RunRouteOptimizer()
    d = app.run(pt1, pt2, target_dist_deg)

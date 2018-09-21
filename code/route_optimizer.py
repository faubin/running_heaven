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

                # cost function increases as run ends and is directed towards
                # the end point, has lower cost towards the end point
                # r_factor should be between 0 (start, new_dist=0) and
                # 1 (end, new_dist=target_d) if new_dist < target_d:
                if new_dist < target_d:
                    r_factor = (target_d - new_dist) / target_d
                else:
                    r_factor = 1.
                cost_dist = (1. - r_factor)**2
                cost_dist *= ((1. + pl.cos(pl.pi+theta_diff))/2.)**2

                # tree weight
                cost_tree = (r_factor)**2 * (temp[2]['tree_density_weight'])**2
                # less cost for routes towards parks
                cost_park = (r_factor)**2 * temp[2]['park_weight']**2
                cost_terms = [cost_dist, cost_tree, cost_park]

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

    def define_park_weight(self, df1, df2, target_d):
        """
        """
        ret = []
        for i in range(len(df1.index)):
            # mean location of the segment
            lon1, lat1 = names.name_to_lon_lat(df1['vertex_start'].iloc[i])
            lon2, lat2 = names.name_to_lon_lat(df1['vertex_end'].iloc[i])
            lon = pl.mean([lon1, lon2])
            lat = pl.mean([lat1, lat2])

            # distance park - segment
            d_to_parks = angles.ang_dist(lon, lat,
                                         angles.rad_to_deg(df2['rep_x_rad']),
                                         angles.rad_to_deg(df2['rep_y_rad']))

            #  weight is quadratic from the park with value of 1 at target
            # distance with 0 at park
            weight = (d_to_parks / target_d)**0.5  # **2
            # select closest park
            weight = min(weight)
            # calue is topped to 1
            if weight > 1.:
                weight = 1.
            ret.append(weight)
        return ret

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
        for i in intersection_names:
            lon, lat = names.name_to_lon_lat(i)
            pl.plot(lon, lat, '.k')

        # plotting the route
        new_gdf2 = gpd.GeoDataFrame(new_df2)
        new_gdf2_path = new_gdf2.iloc[pl.array(path_indices)]
        new_gdf2_path.plot(ax=ax2, color='k', linewidth=4)
        pl.savefig('path_run.png')
        pl.savefig('../app/flaskexample/static/path_run.png')

        return new_gdf2_path["distance"].sum()



    def run(self, pt1, pt2, target_dist_deg):
        """
        """
        new_df2 = gpd.read_file('route_connections.geojson')
        new_df2.rename(index=str, columns={'vertex_sta': 'vertex_start',
                                           'tree_numbe': 'tree_number'},
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

        print('=============== Use density ===============')
        tree_density_norm = 1. - new_df2['tree_number'] / max(new_df2['tree_number'])

        # load data for plotting
        dfs = {}
        for key_ in ['park', 'street', 'sidewalk']:
            dfs[key_] = gpd.read_file('{0:s}_processed.geojson'.format(key_))
        dfs['tree'] = pd.read_csv('tree_processed.csv')
        park_weight = self.define_park_weight(new_df2, dfs['park'],
                                              target_dist_deg)

        # distribution of features for debugging
        # self.feature_distributions(tree_density_norm, park_weight)

        # problem information for Dijkstra's algorithm
        edges = []
        for i in range(len(new_df2.index)):
            # distance - shortest path
            edges.append((new_df2['vertex_start'].iloc[i],  # starting point
                          new_df2['vertex_end'].iloc[i],  # end point
                          0.,  # cost, updated automatically
                          {'distance': new_df2['distance'].iloc[i],
                           'tree_density_weight': tree_density_norm.iloc[i],
                           'park_weight': park_weight[i],
                           }
                          ))

        # optimizing path
        opt_path = self.dijkstra(edges, start_point, end_point,
                                 target_dist_deg)
        print(opt_path)

        # plot the route (straight lines instead of actual geometry)
        xx = copy.deepcopy(opt_path[1])
        prev_pt = None
        path_indices = []
        try:
            while True:
                pl.plot(float(xx[0].split('_')[0]), float(xx[0].split('_')[1]),
                        'ok')
                if prev_pt is not None:
                    ind = new_df2["vertex_start"] == prev_pt
                    ind &= new_df2['vertex_end'] == xx[0]
                    ind = pl.where(ind)[0]
                    if len(ind) > 1:
                        print('problem with selecting path', ind)
                    path_indices.append(ind[0])

                prev_pt = copy.deepcopy(xx[0])
                xx = copy.deepcopy(xx[1])
        except IndexError:
            pass
        print(path_indices)

        # plotting the data and route
        d_path = self.plot_route(dfs, intersection_names, start_point,
                                 end_point, new_df2, path_indices)

        # resulting distance
        print('Total distance is : {0:f} degrees'.format(d_path))
        print('Taget distance was: {0:f} degrees'.format(target_dist_deg))

        if self.show:
            pl.show()

        return d_path

if __name__ == "__main__":
    # pt1, pt2 = (-73.978, 40.778, -73.967, 40.767)
    pt1, pt2 = ('-73.967_40.763', '-73.963_40.772')

    target_dist_deg = 0.011  # shortest distance east of Central Park
    target_dist_deg += 0.005

    app = RunRouteOptimizer()
    d = app.run(pt1, pt2, target_dist_deg)

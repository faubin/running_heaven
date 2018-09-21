from collections import defaultdict
import heapq
import copy
import pandas as pd
import geopandas as gpd
import pylab as pl
import names
import angles
import locations
import build_data_source as bd
import pdb


class RunRouteOptimizer():
    """
    """
    def __init__(self, borough='M'):
        """
        """
        return


    def update_costs(self, object_, target_d, d_done, current_point, end_point):
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
        theta_end = locations.get_angle(lon_current, lat_current, lon_end, lat_end)

        for key_ in [current_point]:
        #for key_ in object_.keys():
            for i in range(len(object_[key_])):
                temp = list(object_[key_][i])
                lon, lat = names.name_to_lon_lat(object_[key_][i][1])

                # angular distance to end
                ang_dist = angles.angular_distance(lon_end, lat_end, lon, lat)

                # cost function
                new_dist = temp[2]['distance'] + d_done[current_point]
                theta_pt = locations.get_angle(lon_current, lat_current, lon, lat)
                theta_diff = theta_end - theta_pt

                # cost function increases as run ends and is directed towards the
                # end point, has lower cost towards the end point
                # r_factor should be between 0 (start, new_dist=0) and
                # 1 (end, new_dist=target_d) if new_dist < target_d:
                if new_dist < target_d:
                    r_factor = (target_d - new_dist) / target_d
                else:
                    r_factor = 1.
                cost_terms = [(1. - r_factor)**2 * ((1. + pl.cos(pl.pi+theta_diff))/2.)**2]
                # tree weight
                cost_terms.append((r_factor)**2 * (temp[2]['tree_density_weight'])**2)
                # less cost for routes towards parks
                cost_terms.append((r_factor)**2 * temp[2]['park_weight']**2)
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

        q, seen, mins_cost, calc_dists = [(0, start_label, (), 0)], set(), {start_label: 0}, {start_label: 0}
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
            d_to_parks = angles.angular_distance(lon, lat,
                                                 angles.rad_to_deg(df2['rep_x_rad']),
                                                 angles.rad_to_deg(df2['rep_y_rad']))
            #dd = []
            #for j in range(len(df2.index)):
            #    geom = df2['geometry'].iloc[j]
            #    dd.append(angles.angular_distance(lon, lat,
            #                                      geom.representative_point().x,
            #                                      geom.representative_point().y))
#           # d_to_parks = angles.angular_distance(lon, lat,
#           #                                      geom.representative_point().x,
#           #                                      geom.representative_point().y)
            #d_to_parks = pl.array(dd)

            # weight is quadratic from the park with value of 1 at target
            # distance with 0 at park
            weight = (d_to_parks / target_d)**0.5#**2
            # select closest park
            weight = min(weight)
            # calue is topped to 1
            if weight > 1.:
                weight = 1.
            ret.append(weight)
        return ret

    def run(self):
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

        #pt1 = locations.get_closest_point_to(-73.978, 40.778, intersection_names)
        #pt2 = locations.get_closest_point_to(-73.967, 40.767, intersection_names)
        pt1 = locations.get_closest_point_to(-73.967, 40.763, intersection_names)
        pt2 = locations.get_closest_point_to(-73.963, 40.772, intersection_names)
        #pt2 = locations.get_closest_point_to(-73.966, 40.765, intersection_names)

        start_point = pt1
        end_point = pt2
        print(pt1, pt2)

        target_dist_deg = 0.011  # shortest distance east of Central Park
        target_dist_deg += 0.005

        tree_density_norm = new_df2['tree_number'] / max(new_df2['tree_number'])

        dfs = {}
        for key_ in ['park', 'street', 'sidewalk']:
            dfs[key_] = gpd.read_file('{0:s}_processed.geojson'.format(key_))  # should be geojson
        dfs['tree'] = pd.read_csv('tree_processed.csv')
        #park_df = gpd.read_file('../raw_data/Parks Zones.geojson')
        #appp = bd.DataBuilder()
        #park_df = appp.zoom_on_data({'park': park_df}, -73.97, 40.77, 0.01)
        #park_df = park_df['park']
        park_weight = self.define_park_weight(new_df2, dfs['park'], target_dist_deg)

        pl.figure(1)#, figsize=(12, 12))
        pl.subplot(221)
        pl.hist(tree_density_norm)
        pl.subplot(222)
        pl.hist(park_weight)

        edges = []
        for i in range(len(new_df2.index)):
            # distance - shortest path
            edges.append((new_df2['vertex_start'].iloc[i],  # starting point
                          new_df2['vertex_end'].iloc[i],  # end point
                          0.,  # cost, updated automatically
                          {'distance': new_df2['distance'].iloc[i],  # distance of segment
                           'tree_density_weight': 1. - tree_density_norm.iloc[i],  # average tree density on the segment
                           'park_weight': park_weight[i],  # normarlized park gradient
                          }
                        ))

        xxx = self.dijkstra(edges, start_point, end_point, target_dist_deg)
        print(xxx)


        # 73.966813_40.764105
        # plotting the selected data
        fig, ax2 = pl.subplots(figsize=(12, 9))
        colors = 'yrb'
        for n_key, key_ in enumerate(['park', 'street', 'sidewalk']):
            dfs[key_].plot(ax=ax2, color=colors[n_key])
        pl.plot(dfs['tree']['longitude'], dfs['tree']['latitude'], '.g', markersize=2)
        #borough_df1.plot(ax=ax2, color='y', label='x')  # roads
        #borough_df2.plot(ax=ax2, color='b', label='y')  # sidewalks
        #borough_df3.plot(ax=ax2, color='r', label='z')  # parks
        #borough_df4.plot('longitude', 'latitude', kind='scatter', ax=ax2)

        # black dot on all intersections
        for i in intersection_names:
            lon, lat = names.name_to_lon_lat(i)
            pl.plot(lon, lat, '.k')

        # plots starting and end point
        lon_start, lat_start = names.name_to_lon_lat(start_point)
        pl.plot(lon_start, lat_start, 'sc', markersize=12)
        lon_end, lat_end = names.name_to_lon_lat(end_point)
        pl.plot(lon_end, lat_end, 'oc', markersize=12)


        # plot the route (straight lines instead of actual geometry)
        xx = copy.deepcopy(xxx[1])
        prev_pt = None
        path_indices = []
        try:
            while True:
                pl.plot(float(xx[0].split('_')[0]), float(xx[0].split('_')[1]), 'ok')
                if prev_pt is not None:
                    ind = pl.where((new_df2["vertex_start"] == prev_pt) & (new_df2['vertex_end'] == xx[0]))[0]
                    if len(ind) > 1:
                        print('problem with selecting path', ind)
                    path_indices.append(ind[0])

                prev_pt = copy.deepcopy(xx[0])
                xx = copy.deepcopy(xx[1])
        except IndexError:
            pass

        pl.xlabel('Longitude ($^o$)', fontsize=20)
        pl.ylabel('Latitude ($^o$)', fontsize=20)
        pl.xticks(fontsize=16)
        pl.yticks(fontsize=16)
        pl.title('NYC Map of Running Areas', fontsize=20)
        #pl.legend(['roads', 'sidewalks', 'parks'], loc=2, prop={'size': 14})

        # make sure I have all the routes
        new_gdf2 = gpd.GeoDataFrame(new_df2)
        print(path_indices)
        new_gdf2_path = new_gdf2.iloc[pl.array(path_indices)]
        new_gdf2_path.plot(ax=ax2, color='k', linewidth=4)
        # new_gdf2.plot(ax=ax2, color='w')
        #pl.savefig('path_run.png')
        print('Total distance is: {0:f} degrees, the taget was {1:f} degrees'.format(new_gdf2_path["distance"].sum(), target_dist_deg))

        pl.savefig('path_run.png')
        pl.savefig('../app/flaskexample/static/path_run.png')
        pl.show()
        return

if __name__ == "__main__":
    app = RunRouteOptimizer()
    app.run()


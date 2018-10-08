#!/usr/bin/env
"""
This class processes the raw data for route_optimizer.py
"""
#! /usr/bin/env python
import ast
import os
import pdb
import time
import requests
import pylab as pl
import geopandas as gpd
import pandas as pd
from shapely import geometry
from running_heaven.code.lib import angles
from running_heaven.code.lib import core
##  from running_heaven.code.lib import data_handler
from running_heaven.code.lib import names
from running_heaven.code.lib import map_plotter


class DataBuilder(core.HeavenCore):
    """
    This class processes the raw data for route_optimizer.py
    """
    def __init__(self, borough='M'):
        core.HeavenCore.__init__(self)
        self.boroughs = {'M': '1',  # Manhattan
                         'X': '2',  # Bronx
                         'B': '3',  # Brooklyn
                         'Q': '4',  # Queens
                         'R': '5'}  # Staten Island
        self.borough_names = {'M': 'Manhattan',
                              'X': 'Bronx',
                              'B': 'Brooklyn',
                              'Q': 'Queens',
                              'R': 'Staten Island'}
        self.borough_name = self.borough_names[borough]
        self.borough_code = self.boroughs[borough]
        self.borough = borough

        # data structure
        if 'data' not in os.listdir(self.running_heaven_path):
            os.mkdir(os.path.join(self.running_heaven_path, 'data'))
        return

    def load_raw_data(self):
        """
        """
        path = os.path.join(self.running_heaven_path, 'raw_data/')
        data_file_names = {'park': 'Parks Zones.geojson',
                           'sidewalk': 'Sidewalk Centerline.geojson',
                           'street': 'NYC Street Centerline (CSCL).geojson'}
        raw_dfs = {}
        for key_ in data_file_names.keys():
            print('Loading {0:s} data'.format(key_))
            raw_dfs[key_] = gpd.read_file(os.path.join(path,
                                                       data_file_names[key_]))
        print('Loading tree data')
        raw_dfs['tree'] = self.load_tree_data()
        print('Done loading data')
        return raw_dfs

    def load_tree_data(self, query_limit=25000):
        """
        To do: load all trees iteratively, and add them to the proper segment
        """
        # download trees from SODA API (
        #     https://dev.socrata.com/consumers/getting-started.html)
        # see https://dev.socrata.com/docs/queries/offset.html for how to
        #     offset and get the whole list
        # query_limit = 25000  # absolute limit is 50 000

        # query loop until receiving all the trees
        i = 0
        done = False
        tree_list = []
        while not done:
            # request
            request = 'https://data.cityofnewyork.us/resource/5rq2-4hqu.json'
            request += '?boroname={0:s}'.format(self.borough_name)
            request += '&$limit={0:d}'.format(query_limit)
            request += '&$offset={0:d}'.format(i*query_limit)
            request_value = requests.get(request)
            text = request_value.text[:-1]

            # convert string to list of dictionary safely
            try:
                query_result_list = ast.literal_eval(text)
            except ValueError:
                exit('There was an error downloading the tree data. Retry.')
            tree_list += query_result_list

            # when less tree than the query limit, we're done
            if not len(query_result_list) == query_limit:
                done = True
            i += 1
            time.sleep(5)

        # convert to DataFrame
        tree_component = pd.DataFrame(tree_list)

        # write data to file
        str_to_float_list = ['longitude', 'latitude']
        tree_component[str_to_float_list] = tree_component[str_to_float_list].astype(float)
        return tree_component

    def get_tree_density(self, geom, tree_component):
        """
        """
        x0 = geom.representative_point().x
        y0 = geom.representative_point().y
        # x1 = geom.boundary[0].x
        # y1 = geom.boundary[0].y
        # x2 = geom.boundary[1].x
        # y2 = geom.boundary[1].y
        # r2 = ((x2-x1)**2 + (y2-y1)**2) / 4.
        r2 = 0.0008212487130291317**2

        inside = (tree_component['longitude'] - x0)**2 + (tree_component['latitude']-y0)**2 <= r2
        return inside.sum()

    def select_data_for_borough(self, map_components):
        """
        only keepds the data in a borough for streets and parks
        for sidewalks, there is no borough information in the data
        so the sidewalks within some (lon, lat) are kept -> not ideal...
        """
        map_components['park'] = map_components['park'].loc[map_components['park']['borough'] == self.borough]
        map_components['street'] = map_components['street'].loc[map_components['street']['borocode'] == self.borough_code]

        # load borough boundary
        boundary_file_name = os.path.join(self.running_heaven_path, 'raw_data',
                                          'Borough Boundaries.geojson')
        borough_limits = gpd.read_file(boundary_file_name)
        is_borough_valid = borough_limits['boro_name'].values == self.borough_name
        borough_geometry = borough_limits.iloc[is_borough_valid]['geometry'].iloc[0]

        # identify the sidewalks outside of the borough
        index_to_reject = []
        for i in map_components['sidewalk'].index:
            lon_deg = angles.rad_to_deg(map_components['sidewalk']['rep_x_rad'].iloc[i])
            lat_deg = angles.rad_to_deg(map_components['sidewalk']['rep_y_rad'].iloc[i])
            point = geometry.Point(lon_deg, lat_deg)
            if not borough_geometry.contains(point):
                index_to_reject.append(i)

        map_components['sidewalk'].drop(index_to_reject, inplace=True)
        map_components['sidewalk'].reset_index(drop=True, inplace=True)

        return map_components

    def add_geometry_representative(self, map_components):
        """
        adds columns to geometry DataFrames with a representative point in rad
        """
        for segment_type, map_component in map_components.items():
            if not segment_type == 'tree':
                rep_x_rad = [angles.deg_to_rad(map_component['geometry'].iloc[i].representative_point().x) for i in range(len(map_component.index))]
                map_component['rep_x_rad'] = pd.Series(rep_x_rad,
                                                       index=map_component.index)
                rep_y_rad = [angles.deg_to_rad(map_component['geometry'].iloc[i].representative_point().y) for i in range(len(map_component.index))]
                map_component['rep_y_rad'] = pd.Series(rep_y_rad,
                                                       index=map_component.index)
        return map_components

    def select_data(self, map_component, lon0_deg, lat0_deg, r_deg,
                    exclude_data):
        """
        drops the data outside a radius r_deg around (lon0_deg, lat0_deg)
        """

        lon0 = angles.deg_to_rad(pl.float64(lon0_deg))  # lambda
        lat0 = angles.deg_to_rad(pl.float64(lat0_deg))  # phi

        # rounding could be an issue...
        rep_x_rad = map_component['rep_x_rad']
        rep_y_rad = map_component['rep_y_rad']
        map_component['diff_to_ref_rad'] = angles.ang_dist(lon0,
                                                           lat0,
                                                           rep_x_rad,
                                                           rep_y_rad)

        if exclude_data:
            diff_to_ref_rad = map_component['diff_to_ref_rad']
            invalid = angles.rad_to_deg(diff_to_ref_rad) > r_deg
            map_component.drop(map_component.index[invalid], inplace=True)
        return map_component

    def select_data_pts(self, map_component, lon0_deg, lat0_deg, r_deg,
                        exclude_data):
        """
        drops the data outside a radius r_deg around (lon0_deg, lat0_deg)
        """
        lon0 = angles.deg_to_rad(pl.float64(lon0_deg))  # lambda
        lat0 = angles.deg_to_rad(pl.float64(lat0_deg))  # phi

        # rounding could be an issue...
        longitude_rad = angles.deg_to_rad(map_component['longitude'])
        latitude_rad = angles.deg_to_rad(map_component['latitude'])
        map_component['diff_to_ref_rad'] = angles.ang_dist(lon0,
                                                           lat0,
                                                           longitude_rad,
                                                           latitude_rad)

        if exclude_data:
            diff_to_ref_rad = map_component['diff_to_ref_rad']
            invalid = angles.rad_to_deg(diff_to_ref_rad) > r_deg
            map_component.drop(map_component.index[invalid], inplace=True)
        return map_component

    def zoom_on_data(self, map_components, lon0_deg, lat0_deg, r_deg, exclude_data):
        """
        """
        for key_ in map_components.keys():
            if key_ == 'tree':
                map_components[key_] = self.select_data_pts(map_components[key_], lon0_deg, lat0_deg, r_deg, exclude_data)
            else:
                map_components[key_] = self.select_data(map_components[key_], lon0_deg, lat0_deg, r_deg, exclude_data)
        return map_components

    def plot_data(self, map_components):
        """
        """
        # plotting the selected data
        ax1 = pl.subplots(figsize=(12, 12))[1]
        colors = {'park': 'y', 'street': 'r', 'sidewalk': 'b'}
        for key_ in colors.keys():
            map_components[key_].plot(ax=ax1, color=colors[key_])
        pl.plot(map_components['tree']['longitude'], map_components['tree']['latitude'],
                '.g', markersize=2)

        pl.xlabel('Longitude ($^o$)', fontsize=20)
        pl.ylabel('Latitude ($^o$)', fontsize=20)
        pl.xticks(fontsize=16)
        pl.yticks(fontsize=16)
        pl.title('NYC Map of Running Areas', fontsize=20)
        pl.savefig('data_central_park.png')
        pl.show()
        return

    def extract_segments_from_df(self, map_components):
        """
        """
        # extract the important information from the segments
        data_for_df = {'lon_start': [], 'lat_start': [], 'lon_end': [],
                       'lat_end': [], 'distance': [], 'type': [],
                       'connections_start': [], 'connections_end': [],
                       'name_start': [], 'name_end': [],
                       'geometry': [], 'tree_number': []}
        print('Warning: radius for tree is hardcoded in get_tree_density().')
        for key_ in ['sidewalk', 'street']:
            for i in range(len(map_components[key_].index)):
                geom = map_components[key_]['geometry'].iloc[i]
                try:
                    data_for_df['lon_start'].append(geom.boundary[0].x)
                except IndexError:
                    print('skipping', key_, i)
                    continue
                data_for_df['lat_start'].append(geom.boundary[0].y)
                data_for_df['lon_end'].append(geom.boundary[1].x)
                data_for_df['lat_end'].append(geom.boundary[1].y)
                data_for_df['distance'].append(geom.length)
                data_for_df['type'].append(key_)
                data_for_df['connections_start'].append([])
                data_for_df['connections_end'].append([])
                start_label = names.lon_lat_to_name(data_for_df['lon_start'][-1], data_for_df['lat_start'][-1])
                data_for_df['name_start'].append(start_label)
                end_label = names.lon_lat_to_name(data_for_df['lon_end'][-1],
                                                  data_for_df['lat_end'][-1])
                data_for_df['name_end'].append(end_label)
                data_for_df['geometry'].append(geom)
                tree_number = self.get_tree_density(geom, map_components['tree'])
                data_for_df['tree_number'].append(tree_number)

        for i in range(len(data_for_df['lon_start'])):
            for j in range(i+1, len(data_for_df['lon_start'])):
                if data_for_df["lon_start"][i] == data_for_df["lon_start"][j] and data_for_df["lat_start"][i] == data_for_df["lat_start"][j]:
                    data_for_df['connections_start'][i].append(j)
                    data_for_df['connections_start'][j].append(i)
                if data_for_df["lon_start"][i] == data_for_df["lon_end"][j] and data_for_df["lat_start"][i] == data_for_df["lat_end"][j]:
                    data_for_df['connections_start'][i].append(j)
                    data_for_df['connections_end'][j].append(i)
                if data_for_df["lon_end"][i] == data_for_df["lon_start"][j] and data_for_df["lat_end"][i] == data_for_df["lat_start"][j]:
                    data_for_df['connections_end'][i].append(j)
                    data_for_df['connections_start'][j].append(i)
                if data_for_df["lon_end"][i] == data_for_df["lon_end"][j] and data_for_df["lat_end"][i] == data_for_df["lat_end"][j]:
                    data_for_df['connections_end'][i].append(j)
                    data_for_df['connections_end'][j].append(i)

        return data_for_df

    def convert_segments_to_vertex(self, data):
        """
        Given the list of segments and their connections, provides
        vertex-to-vertex information
        """
        vertices = {'vertex_start': [], 'vertex_end': [], 'distance': [],
                    'type': [], 'geometry': [], 'tree_number': [], }
#                     'park_weight': []}

        # loop over all the segments
        for i in range(len(data['type'])):
            # loop over the 2 extremities of each segment
            for which in ['start', 'end']:
                name1 = names.lon_lat_to_name(data['lon_'+which][i],
                                              data['lat_'+which][i])

                # loop over the connected segments
                for j in data['connections_'+which][i]:
                    # find the vertices who are the same to identify which
                    # distance to keep
                    if data['name_start'][j] == name1:
                        name2 = data['name_end'][j]
                    elif data['name_end'][j] == name1:
                        name2 = data['name_start'][j]
                    else:
                        raise ValueError('Problem!!!')

                    # find if the vertex is already defined
                    is_defined1 = pl.array(vertices["vertex_start"]) == name1
                    is_defined2 = pl.array(vertices['vertex_end']) == name2
                    is_definedx = pl.logical_and(is_defined1, is_defined2)

                    # find if the vertex is already defined
                    is_defined1 = pl.array(vertices["vertex_start"]) == name2
                    is_defined2 = pl.array(vertices['vertex_end']) == name1
                    is_definedy = pl.logical_and(is_defined1, is_defined2)
                    is_defined = pl.logical_or(is_definedx, is_definedy)
                    is_defined = len(pl.where(is_defined)[0])

                    # stores the path informations
                    if not is_defined:
                        vertices['vertex_start'].append(name1)
                        vertices['vertex_end'].append(name2)
                        vertices['distance'].append(data['distance'][j])
                        vertices['tree_number'].append(data['tree_number'][j])
                        vertices['type'].append(data['type'][j])
                        vertices['geometry'].append(data['geometry'][j])
        # === write function to add tree numbers
        return vertices

    def write_data_to_file(self, connections_dict):
        """
        """
        columns = ['vertex_start', 'vertex_end', 'distance', 'type',
                   'geometry', 'tree_number', 'tree_density',
                   'min_dist_to_park']
        connections_pd = pd.DataFrame(connections_dict, columns=columns)
        connections_gpd = gpd.GeoDataFrame(connections_pd)
        connections_gpd.to_file(os.path.join(self.running_heaven_path,
                                             'data',
                                             'processed',
                                             'route_connections.geojson'))

    def write_processed_data_to_file(self, map_components):
        """
        write data to file for plotting when optimizing routes
        """
        if 'processed' not in os.listdir(os.path.join(self.running_heaven_path, 'data')):
            os.mkdir(os.path.join(self.running_heaven_path, 'data',
                                  'processed'))
        for key_ in map_components.keys():
            if key_ == 'tree':
                file_name = os.path.join(self.running_heaven_path, 'data',
                                         'processed', '{0:s}.csv'.format(key_))
                map_components[key_].to_csv(file_name)
            else:
                file_name = os.path.join(self.running_heaven_path, 'data',
                                         'processed',
                                         '{0:s}.geojson'.format(key_))
                map_components[key_].to_file(file_name)
        return

    def add_weights(self, dict_, map_components):
        """
        """
        is_sidewalk = pl.array(dict_['type']) == 'sidewalk'
        is_street = pl.array(dict_['type']) == 'street'

        # relative tree density
        dict_['tree_density'] = pl.array(dict_['tree_number']).astype(float)
        dict_['tree_density'] /= pl.array(dict_['distance'])

        # sidewalks wet to maximum
        # max_density = max(dict_['tree_density'])
        # dict_['tree_density'][is_sidewalk] = max_density

        # dict_['tree_density'] /= max(dict_['tree_density'])
        dict_['tree_density'] /= pl.percentile(dict_['tree_density'][is_street], 85.)
        pl.hist(dict_['tree_density'], bins=pl.arange(0., 1.5, 0.05))

        # randomize weight of sidewalks
        dict_['tree_density'][is_sidewalk] = 0.75+0.15*pl.randn(len(dict_['tree_density'][is_sidewalk]))
        dict_['tree_density'][dict_['tree_density'] > 1.] = 1.
        # randomize streets with 0 trees
        zero_tree = pl.array(dict_['tree_number']) == 0
        dict_['tree_density'][zero_tree] = 0.25+0.15*pl.randn(sum(zero_tree))

        # tree density defined as 1 in parks since no tree data there
        dict_['tree_density'][dict_['tree_density'] > 0.999] = 0.999
        dict_['tree_density'][dict_['tree_density'] < 0.] = 0.
        pl.hist(dict_['tree_density'], bins=pl.arange(0., 1.5, 0.05))

        dict_['min_dist_to_park'] = []
        for i in range(len(dict_['vertex_start'])):
            # mean location of the segment
            lon = dict_['geometry'][i].representative_point().x
            lat = dict_['geometry'][i].representative_point().y

            # distance park - segment
            dist = angles.ang_dist(lon, lat,
                                   angles.rad_to_deg(map_components['park']['rep_x_rad']),
                                   angles.rad_to_deg(map_components['park']['rep_y_rad']))
            dist = min(dist)
            dict_['min_dist_to_park'].append(dist)
        return dict_

    def run(self):
        """
        Takes 20 minutes for Manhattan
        """
        raw_map_components = self.load_raw_data()
        map_components = self.add_geometry_representative(raw_map_components)
        data_dfs = self.select_data_for_borough(map_components)

        # data_hand = data_handler.DataHandler()
        map_plotter.plot_raw_data(data_dfs)

        # include only data around central park for debugging
        # data_dfs = self.zoom_on_data(data_dfs, -73.97, 40.77, 0.01, False)
        # central park, small
        # data_dfs = self.zoom_on_data(data_dfs, -73.97, 40.77, 0.01, True)
        # central park, big
        # data_dfs = self.zoom_on_data(data_dfs, -73.97, 40.77, 0.02, True)

        # data_dfs = self.zoom_on_data(data_dfs, -73.994, 40.740, 0.01, False)
        # data_dfs = self.zoom_on_data(data_dfs, -73.994, 40.740, 0.01, True)
        # self.plot_data(data_dfs)

        # south manhattan???
        # data_dfs = self.zoom_on_data(data_dfs, -73.99, 40.73, 0.02, True)

        # need this to run to add features to the dfs
        # adds the rep_x_rad fields
        # data_dfs = self.zoom_on_data(data_dfs, -73.97, 40.77, 1., False)

        self.write_processed_data_to_file(data_dfs)

        print('Converting data to segments')
        data_dict = self.extract_segments_from_df(data_dfs)
        print('Getting connections from segments')
        conn_dict = self.convert_segments_to_vertex(data_dict)
        print('Adding weights')
        connections_dict = self.add_weights(conn_dict, data_dfs)
        print('writing connections to file')
        self.write_data_to_file(connections_dict)

        # checking for duplicates
        for i in range(len(conn_dict['vertex_start'])):
            for j in range(i+1, len(conn_dict['vertex_start'])):
                if conn_dict['vertex_start'][i] == conn_dict['vertex_start'][j]:
                    if conn_dict['vertex_end'][i] == conn_dict['vertex_end'][j]:
                        print(i, j)
                if conn_dict['vertex_start'][i] == conn_dict['vertex_end'][j]:
                    if conn_dict['vertex_end'][i] == conn_dict['vertex_start'][j]:
                        print(i, j)

        return


if __name__ == "__main__":
    APP = DataBuilder()
    APP.run()

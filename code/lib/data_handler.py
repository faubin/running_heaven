#!/usr/bin/env
"""
This class allows to perform common operation on data, such as loading and
plotting.
"""
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from running_heaven.code.lib import core


class DataHandler(core.HeavenCore):
    """
    A class to interface with the Google Map API
    """
    def __init__(self):
        core.HeavenCore.__init__(self)

    def load_raw_data(self):
        """
        Load the data as pd.DataFrames

        The return is a dictionary of pd.DataFrames with keys ['street',
        'park', 'sidewalk', 'tree']
        """
        # load data for plotting
        map_components = {}
        path = os.path.join(self.running_heaven_path, 'data', 'processed')
        for data_name in ['park', 'street', 'sidewalk']:
            full_file_name = os.path.join(path,
                                          '{0:s}.geojson'.format(data_name))
            map_components[data_name] = gpd.read_file(full_file_name)
        map_components['tree'] = pd.read_csv(os.path.join(path, 'tree.csv'))
        return map_components

    def load_processed_data(self):
        """
        Loads the processed data from build_data_source.py

        Output:
            a pd.DataFrame with all the segments of roads
        """
        full_file_path = os.path.join(self.running_heaven_path, 'data',
                                      'processed', 'route_connections.geojson')
        segments = gpd.read_file(full_file_path)
        segments_fixed_column_names = self.fix_processed_column_names(segments)
        return segments_fixed_column_names

    def fix_processed_column_names(self, segments):
        """
        The column names are truncated to 10 caracters. Renaming them properly
        """
        segments.rename(index=str, columns={'vertex_sta': 'vertex_start',
                                            'tree_numbe': 'tree_number',
                                            'tree_densi': 'tree_density',
                                            'min_dist_t': 'min_dist_to_park'},
                        inplace=True)
        return segments

    def find_index_of_nearest_segment(self, segments, gps_route):
        """
        given a GPS route, finds the closest segments in the database

        Inputs:
            segments is a pd.FataFrame
            gps_route is a pd.DataFrame with 2 columns: Longitude and Latitude
        Output:
            selected_segments is a pd.DataFrame with only the segments closest
                to the GPS route
        """
        # convert the processed data to a individual points
        point_list = []
        index_list = []
        lon_list = []
        lat_list = []
        for index, row in segments.iterrows():
            lon = row['geometry'].xy[0]
            lat = row['geometry'].xy[1]
            for llon, llat in zip(lon, lat):
                point_list.append(shapely.geometry.Point(llon, llat))
                index_list.append(int(index))
                lon_list.append(llon)
                lat_list.append(llat)
        multi_pts = shapely.geometry.MultiPoint(point_list)

        # loop over all the points in the gps
        selected_index = []
        for i in gps_route.index.values:
            gps_pt = shapely.geometry.Point(gps_route['Longitude'].iloc[i],
                                            gps_route['Latitude'].iloc[i])
            nearest_geoms = shapely.ops.nearest_points(gps_pt, multi_pts)
            # closest_index = nearest_geoms.index(nearest_geoms[1])
            # print(pt, closest_index)
            lon_closest = nearest_geoms[1].xy[0][0]
            lat_closest = nearest_geoms[1].xy[1][0]

            index = np.logical_and(np.array(lon_list) == lon_closest,
                                   np.array(lat_list) == lat_closest)
            selected_index.append(np.where(index)[0][0])

        unique_index_list = set(np.array(index_list)[np.array(selected_index)])
        selected_segments = segments.iloc[sorted([i for i in unique_index_list])]

        return selected_segments


if __name__ == "__main__":
    APP = DataHandler()
    APP.load_raw_data()

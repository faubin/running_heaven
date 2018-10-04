import geopandas as gpd
import numpy as np
import os
import pandas as pd
import pdb
try:
    import pylab as pl
except ImportError:
    print('Warning: matplolib failed to load')
from running_heaven.code import names
from running_heaven.code import core


class DataHandler(core.HeavenCore):
    """
    A class to interface with the Google Map API
    """
    def __init__(self):
        core.HeavenCore.__init__(self)

    def load_raw_data(self):
        """
        """
        # load data for plotting
        dfs = {}
        path = os.path.join(self.running_heaven_path, 'data', 'processed')
        for data_name in ['park', 'street', 'sidewalk']:
            full_file_name = os.path.join(path,
                                          '{0:s}.geojson'.format(data_name))
            dfs[data_name] = gpd.read_file(full_file_name)
        dfs['tree'] = pd.read_csv(os.path.join(path, 'tree.csv'))
        return dfs

    def plot_raw_data(self, dfs, xlim=None, ylim=None, dim_colors=False,
                      show=False):
        """
        """
        if dim_colors:
            colors = {'park': '#ffff80','street': '#ff9980',
                      'sidewalk': '#8080ff', 'tree': '#009933'}
        else:
            colors = {'park': '#e6e600', 'street': '#ff3300',
                      'sidewalk': '#0000ff', 'tree': '#009933'}

        # plotting the map
        fig, ax = pl.subplots(figsize=(12, 9))
        for n_key, data_type in enumerate(dfs.keys()):
            if data_type not in colors.keys():
                raise ValueError('The color for plotting {0:s} is not \
                                  defined'.format(data_type))
            if data_type == 'tree':
                dfs[data_type].plot.scatter('longitude', 'latitude', 2,
                                            colors[data_type], ax=ax)
            else:
                dfs[data_type].plot(ax=ax, color=colors[data_type])
        pl.xlabel('Longitude ($^o$)', fontsize=20)
        pl.ylabel('Latitude ($^o$)', fontsize=20)
        pl.xticks(fontsize=16)
        pl.yticks(fontsize=16)

        if xlim is not None and type(xlim) == list:
            pl.xlim(xlim)
        if ylim is not None and type(ylim) == list:
            pl.ylim(ylim)

        return ax

    def plot_raw_data_step_by_step(self, dfs, xlim=None, ylim=None):
        """
        """
        colors = {'dim': {'park': '#ffff80','street': '#ff9980',
                          'sidewalk': '#8080ff', 'tree': '#009933'},
                  'standard': {'park': '#e6e600', 'street': '#ff3300',
                               'sidewalk': '#0000ff', 'tree': '#009933'}}

        data_types = ['park', 'street', 'sidewalk', 'tree']

        for n_data_type in range(1, len(data_types)+1):

            fig, ax = pl.subplots(figsize=(12, 9))
            for n_iter, data_type in enumerate(data_types[:n_data_type]):
                color_type = 'dim'
                #if n_data_type == n_iter + 1:
                #    color_type = 'standard'
                #else:
                #    color_type = 'dim'

                if data_type not in colors[color_type].keys():
                    raise ValueError('The color for plotting {0:s} is not \
                                      defined'.format(data_type))
                if data_type == 'tree':
                    dfs[data_type].plot.scatter('longitude', 'latitude', 2,
                                                colors[color_type][data_type],
                                                ax=ax)
                else:
                    dfs[data_type].plot(ax=ax,
                                        color=colors[color_type][data_type])
            pl.xlabel('Longitude ($^o$)', fontsize=20)
            pl.ylabel('Latitude ($^o$)', fontsize=20)
            pl.xticks(fontsize=16)
            pl.yticks(fontsize=16)

            if xlim is not None and type(xlim) == list:
                pl.xlim(xlim)
            if ylim is not None and type(ylim) == list:
                pl.ylim(ylim)
            pl.savefig('raw_data_{0:d}.png'.format(n_data_type))
            pl.close()

    def plot_route(self, map_components, start_point, end_point, segments,
                   path_indices):
        """
        map_components is a pd.DataFrame with the raw NYC data
        start_point and end_point are str with the form lon_lat
        segments is a pd.DataFrame with the processed segments
        path indices is a list of the rows in segments that are part of the
        route
        """
        # plotting the map
        ax = self.plot_raw_data(map_components, xlim=[-73.985, -73.955],
                                          ylim=[40.760, 40.785],
                                          dim_colors=True)

        # plots starting and end point
        lon_start, lat_start = names.name_to_lon_lat(start_point)
        pl.plot(lon_start, lat_start, 's', color='#ff33cc', markersize=12)
        lon_end, lat_end = names.name_to_lon_lat(end_point)
        pl.plot(lon_end, lat_end, 'o', color='#ff33cc', markersize=12)

        # plotting the route
        if path_indices is not None:
            segments_gpd = gpd.GeoDataFrame(segments)
            route = segments_gpd.iloc[np.array(path_indices)]
            route.plot(ax=ax, color='k', linewidth=4)
        pl.savefig('path_run.png')
        # pl.savefig('../app/flaskexample/static/path_run.png')
        return

    def load_processed_data(self):
        """
        """
        full_file_path = os.path.join(self.running_heaven_path, 'data',
                                      'processed', 'route_connections.geojson')
        df = gpd.read_file(full_file_path)
        df = self.fix_processed_column_names(df)
        return df

    def fix_processed_column_names(self, df):
        """
        The column names are truncated to 10 caracters. Renaming them properly
        """
        df.rename(index=str, columns={'vertex_sta': 'vertex_start',
                                      'tree_numbe': 'tree_number',
                                      'tree_densi': 'tree_density',
                                      'min_dist_t': 'min_dist_to_park'},
                  inplace=True)
        return df

    def find_index_of_nearest_segment(self, df_proc, df):
        from shapely.geometry import Point
        from shapely.geometry import MultiPoint
        from shapely.ops import nearest_points

        # convert the processed data to a individual points
        point_list = []
        index_list = []
        lon_list = []
        lat_list = []
        for index, row in df_proc.iterrows():
            lon = row['geometry'].xy[0]
            lat = row['geometry'].xy[1]
            for i in range(len(lon)):
                point_list.append(Point(lon[i], lat[i]))
                index_list.append(int(index))
                lon_list.append(lon[i])
                lat_list.append(lat[i])
        multi_pts = MultiPoint(point_list)

        # loop over all the points in the gps
        selected_index = []
        for i in df.index.values:
            pt = Point(df['Longitude'].iloc[i], df['Latitude'].iloc[i])
            nearest_geoms = nearest_points(pt, multi_pts)
            #closest_index = nearest_geoms.index(nearest_geoms[1])
            #print(pt, closest_index)
            lon_closest = nearest_geoms[1].xy[0][0]
            lat_closest = nearest_geoms[1].xy[1][0]

            index = pl.logical_and(pl.array(lon_list) == lon_closest,
                                   pl.array(lat_list) == lat_closest)
            selected_index.append(pl.where(index)[0][0])

        unique_index_list = set(pl.array(index_list)[pl.array(selected_index)])
        new_df = df_proc.iloc[sorted([i for i in unique_index_list])]

        return new_df


if __name__ == "__main__":
    app = DataHandler()
    dfs = app.load_raw_data()
    import pdb;pdb.set_trace()


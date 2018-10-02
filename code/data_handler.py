from running_heaven.code import core
import os
import pylab as pl
import pandas as pd
import geopandas as gpd


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

    def plot_raw_data(self, dfs, xlim=None, ylim=None, show=False):
        """
        """
        colors = {'park': 'y', 'street': 'r', 'sidewalk': 'b', 'tree': '.g'}

        # plotting the map
        fig, ax = pl.subplots(figsize=(12, 9))
        for n_key, data_type in enumerate(dfs.keys()):
            if data_type not in colors.keys():
                raise ValueError('A color need to be defined for plotting')
            if len(colors[data_type]) > 1:
                ax.plot(dfs[data_type]['longitude'],
                        dfs[data_type]['latitude'], colors[data_type],
                        markersize=2)
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


#! /usr/bin/env python
import os
import ast
import geopandas as gpd
import pandas as pd
import pylab as pl
import requests
import angles
import names
import pdb


class DataBuilder():
    """
    """
    def __init__(self, borough='M'):
        """
        """
        self.boroughs = {'M': '1',  # Manhattan
                         '?': '2',  # Bronx
                         '?': '3',  # Brooklyn
                         '?': '4',  # Queens
                         '?': '5'}  # Staten Island
        self.borough_names = {'M': 'Manhattan',
                             '?': 'Bronx',
                             '?': 'Brooklyn',
                             '?': 'Queens',
                             '?': 'Staten Island'}
        self.borough_name = self.borough_names[borough]
        self.borough_code = self.boroughs[borough]
        self.borough = borough
        # missing boroughs are: {'B', 'Q', 'R', 'X'}
        return

    def load_raw_data(self):
        """
        """
        path = '../raw_data/'
        data_file_names = {'park': 'Parks Zones.geojson',
                           'sidewalk': 'Sidewalk Centerline.geojson',
                           'street': 'NYC Street Centerline (CSCL).geojson',
                           }
        raw_dfs = {}
        for key_ in data_file_names.keys():
            print('Loading {0:s} data'.format(key_))
            raw_dfs[key_] = gpd.read_file(os.path.join(path,
                                                       data_file_names[key_]))
        print('Loading tree data')
        raw_dfs['tree'] = self.load_tree_data()
        print('Done loading data')
        return raw_dfs

    def load_tree_data(self):
        """
        To do: load all trees iteratively, and add them to the proper segment
        """
        # download trees from SODA API (
        #     https://dev.socrata.com/consumers/getting-started.html)
        # see https://dev.socrata.com/docs/queries/offset.html for how to
        #     offset and get the whole list
        query_limit = 25000  # absolute limit is 50 000
        request = 'https://data.cityofnewyork.us/resource/5rq2-4hqu.json'
        request += '?boroname={0:s}'.format(self.borough_name)
        request += '&$limit={0:d}'.format(query_limit)
        r = requests.get(request)
        text = r.text[:-1]


        #t concert string to list of dictionary safely
        tree_list = ast.literal_eval(text)
        if len(tree_list) == query_limit:
            print('Warning: number of trees returned {0:d} is the same as the limit {1:d}'.format(len(tree_list), query_limit))
            print('You are likely missing some trees...')
        # convert to DataFrame
        df = pd.DataFrame(tree_list)
        # write data to file
        # borough_df4.to_csv('raw_data/Tree_{0:s}.csv'.format(borough))
        str_to_float_list = ['longitude', 'latitude']
        df[str_to_float_list] = df[str_to_float_list].astype(float)
        return df

    def get_tree_density(self, geom, df):
        """
        """
        x0 = geom.representative_point().x
        y0 = geom.representative_point().y
        x1 = geom.boundary[0].x
        y1 = geom.boundary[0].y
        x2 = geom.boundary[1].x
        y2 = geom.boundary[1].y
        r2 = ((x2-x1)**2 + (y2-y1)**2) / 4.

        inside = (df['longitude'] - x0)**2 + (df['latitude']-y0)**2 <= r2
        return inside.sum()

    def select_data_for_debug(self, dfs):
        """
        """
        dfs['park'] = dfs['park'].loc[dfs['park']['borough'] == self.borough]
        # remove me after debugging
        if 'street' in dfs.keys():
            dfs['street'] = dfs['street'].loc[dfs['street']['borocode'] == self.borough_code]
        return dfs

    def debug_plot(self, dfs):
        """
        """
        fig, ax1 = pl.subplots()#figsize=(12, 12))
        colors = 'ybrgmc'
        for n_key, key_ in enumerate(dfs.keys()):
            dfs[key_].plot(ax=ax1, color=colors[n_key])

        # to do: remove data outside of proper borough
        x_range = [-74.05, -73.9]
        y_range = [40.65, 40.9]
        x0 = pl.mean(x_range)
        y0 = pl.mean(y_range)
        dx = x_range[1] - x_range[0]
        dy = y_range[1] - y_range[0]
        dd = max([dx, dy]) / 2.
        pl.xlim([x0-dd, x0+dd])
        pl.ylim([y0-dd, y0+dd])
        pl.xlabel('Longitude ($^o$)', fontsize=20)
        pl.ylabel('Latitude ($^o$)', fontsize=20)
        pl.xticks(fontsize=16)
        pl.yticks(fontsize=16)
        pl.title('NYC Map of Running Areas', fontsize=20)
        pl.show()
        return

    def select_data(self, df, lon0_deg, lat0_deg, r_deg):
        """
        drops the data outside a radius r_deg around (lon0_deg, lat0_deg)
        """
        df['rep_x_rad'] = pd.Series([angles.deg_to_rad(df['geometry'].iloc[i].representative_point().x) for i in range(len(df.index))], index=df.index)
        df['rep_y_rad'] = pd.Series([angles.deg_to_rad(df['geometry'].iloc[i].representative_point().y) for i in range(len(df.index))], index=df.index)

        lon0 = angles.deg_to_rad(pl.float64(lon0_deg))  # lambda
        lat0 = angles.deg_to_rad(pl.float64(lat0_deg))  # phi

        # rounding could be an issue...
        df['diff_to_ref_rad'] = angles.angular_distance(lon0, lat0, df['rep_x_rad'], df['rep_y_rad'])
        invalid = angles.rad_to_deg(df['diff_to_ref_rad']) > r_deg
        df.drop(df.index[invalid], inplace=True)

        return df

    def select_data_pts(self, df, lon0_deg, lat0_deg, r_deg):
        """
        drops the data outside a radius r_deg around (lon0_deg, lat0_deg)
        """
        lon0 = angles.deg_to_rad(pl.float64(lon0_deg))  # lambda
        lat0 = angles.deg_to_rad(pl.float64(lat0_deg))  # phi

        # rounding could be an issue...
        df['diff_to_ref_rad'] = angles.angular_distance(lon0, lat0, angles.deg_to_rad(df['longitude']), angles.deg_to_rad(df['latitude']))
        invalid = angles.rad_to_deg(df['diff_to_ref_rad']) > r_deg
        df.drop(df.index[invalid], inplace=True)

        return df

    def zoom_on_data(self, dfs, lon0_deg, lat0_deg, r_deg):
        """
        """
        for key_ in dfs.keys():
            if key_ == 'tree':
                dfs[key_] = self.select_data_pts(dfs[key_], lon0_deg, lat0_deg,
                                                 r_deg)
            else:
                dfs[key_] = self.select_data(dfs[key_], lon0_deg, lat0_deg,
                                             r_deg)
        return dfs

    def plot_data(self, dfs):
        """
        """
        # plotting the selected data
        fig, ax1 = pl.subplots(figsize=(12, 12))
        colors = {'park': 'y', 'street': 'r', 'sidewalk': 'b'}
        for n_key, key_ in enumerate(colors.keys()):#enumerate(dfs.keys()):
            dfs[key_].plot(ax=ax1, color=colors[key_])
        pl.plot(dfs['tree']['longitude'], dfs['tree']['latitude'],
                '.g', markersize=2)

        pl.xlabel('Longitude ($^o$)', fontsize=20)
        pl.ylabel('Latitude ($^o$)', fontsize=20)
        pl.xticks(fontsize=16)
        pl.yticks(fontsize=16)
        pl.title('NYC Map of Running Areas', fontsize=20)
        pl.savefig('data_central_park.png')
        pl.show()
        return

    def extract_info_from_df(self, dfs):
        """
        """
        # extract the important information from the segments
        data_for_df = {'lon_start': [], 'lat_start': [], 'lon_end': [],
                       'lat_end': [], 'distance':[], 'type': [],
                       'connections_start': [], 'connections_end': [],
                       'name_start': [], 'name_end': [],
                       'geometry': [], 'tree_number': [],
                       # 'park_weight': [],
                       }
        for key_ in ['sidewalk', 'street']:
            for i in range(len(dfs[key_].index)):
                geom = dfs[key_]['geometry'].iloc[i]
                data_for_df['lon_start'].append(geom.boundary[0].x)
                data_for_df['lat_start'].append(geom.boundary[0].y)
                data_for_df['lon_end'].append(geom.boundary[1].x)
                data_for_df['lat_end'].append(geom.boundary[1].y)
                data_for_df['distance'].append(geom.length)
                data_for_df['type'].append(key_)
                data_for_df['connections_start'].append([])
                data_for_df['connections_end'].append([])
                data_for_df['name_start'].append(names.lon_lat_to_name(data_for_df['lon_start'][-1],
                                                 data_for_df['lat_start'][-1]))
                data_for_df['name_end'].append(names.lon_lat_to_name(data_for_df['lon_end'][-1], data_for_df['lat_end'][-1]))
                data_for_df['geometry'].append(geom)
                data_for_df['tree_number'].append(self.get_tree_density(geom, dfs['tree']))


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
                    'type': [], 'geometry': [], 'tree_number': [],}
                    #'park_weight': []}

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

                    #pdb.set_trace()
                    #is_defined = len(pl.where((pl.array(vertices["vertex_start"]) == name1) & (pl.array(vertices['vertex_end']) == name2))[0])
                    is_defined1 = pl.array(vertices["vertex_start"] == name1)
                    is_defined2 = pl.array(vertices['vertex_end'] == name2)
                    is_defined = pl.logical_and(is_defined1, is_defined2)
                    is_defined = len(pl.where(is_defined)[0])

                    # stores the path informations
                    if not is_defined:
                        vertices['vertex_start'].append(name1)
                        vertices['vertex_end'].append(name2)
                        vertices['distance'].append(data['distance'][j])
                        vertices['tree_number'].append(data['tree_number'][j])
                        vertices['type'].append(data['type'][j])
                        vertices['geometry'].append(data['geometry'][j])
        return vertices

    def write_data_to_file(self, data_dict):
        """
        """
        columns = ['vertex_start', 'vertex_end', 'distance', 'type',
                   'geometry', 'tree_number']
        df = pd.DataFrame(data_dict, columns=columns)
        df = gpd.GeoDataFrame(df)
        df.to_file("route_connections.geojson")
        return

    def run(self):
        """
        """
        raw_data_dfs = self.load_raw_data()
        data_dfs = self.select_data_for_debug(raw_data_dfs)
        # self.debug_plot(data_dfs)
        data_dfs = self.zoom_on_data(data_dfs, -73.97, 40.77, 0.01)  # zoom on central park
        # self.plot_data(data_dfs)

        for key_ in data_dfs.keys():
            if key_ == 'tree':
                data_dfs[key_].to_csv("{0:s}_processed.csv".format(key_))
            else:
                data_dfs[key_].to_file("{0:s}_processed.geojson".format(key_))

        data_dict = self.extract_info_from_df(data_dfs)
        conn_dict = self.convert_segments_to_vertex(data_dict)
        self.write_data_to_file(conn_dict)
        return


if __name__ == "__main__":
    app = DataBuilder()
    app.run()


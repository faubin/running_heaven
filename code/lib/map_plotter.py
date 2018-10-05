#!/usr/bin/env
"""
A library to plot maps and routes
"""
import geopandas as gpd
import numpy as np
try:
    import pylab as pl
except ImportError:
    print('Warning: pylab failed to load')
from running_heaven.code.lib import names


def plot_raw_data(map_components, xlim=None, ylim=None, show=False):
    """
    Plots a map of the raw data.

    Inputs:
        map_components is a dictionary of pd.DataFrames (the output of
            data_handler.load_raw_data())
        xlim and ylim are list to force the axis of the map, default shows
            the whole map
        show is a booleand to show or not the map

    Return:
        the figure handles
    """
    colors = {'park': '#ffff80',
              'street': '#ff9980',
              'sidewalk': '#8080ff',
              'tree': '#009933'}

    # plotting the map
    subplot_axes = pl.subplots(figsize=(12, 9))[1]
    for data_type in map_components.keys():
        if data_type not in colors.keys():
            raise ValueError('The color for plotting {0:s} is not \
                              defined'.format(data_type))
        if data_type == 'tree':
            map_components[data_type].plot.scatter('longitude',
                                                   'latitude',
                                                   2,
                                                   colors[data_type],
                                                   ax=subplot_axes)
        else:
            map_components[data_type].plot(ax=subplot_axes,
                                           color=colors[data_type])
    pl.xlabel('Longitude ($^o$)', fontsize=20)
    pl.ylabel('Latitude ($^o$)', fontsize=20)
    pl.xticks(fontsize=16)
    pl.yticks(fontsize=16)

    if xlim is not None and isinstance(xlim, list):
        pl.xlim(xlim)
    if ylim is not None and isinstance(ylim, list):
        pl.ylim(ylim)
    if show:
        pl.show()
    return subplot_axes


def plot_raw_data_step_by_step(map_components, xlim=None, ylim=None):
    """
    Plot the data layer by layer and saving the figures

    Inputs:
        map_components is a dictionary of pd.DataFrames (the output of
            data_handler.load_raw_data())
        xlim and ylim are list to force the axis of the map, default shows
            the whole map
        show is a booleand to show or not the map
    """
    colors = {'park': '#ffff80',
              'street': '#ff9980',
              'sidewalk': '#8080ff',
              'tree': '#009933'}
    data_types = ['park', 'street', 'sidewalk', 'tree']

    for n_data_type in range(1, len(data_types)+1):

        subplot_axes = pl.subplots(figsize=(12, 9))[1]
        for data_type in data_types[:n_data_type]:
            if data_type not in colors.keys():
                raise ValueError('The color for plotting {0:s} is not \
                                  defined'.format(data_type))
            if data_type == 'tree':
                map_components[data_type].plot.scatter('longitude',
                                                       'latitude',
                                                       2,
                                                       colors[data_type],
                                                       ax=subplot_axes)
            else:
                map_components[data_type].plot(ax=subplot_axes,
                                               color=colors[data_type])
        pl.xlabel('Longitude ($^o$)', fontsize=20)
        pl.ylabel('Latitude ($^o$)', fontsize=20)
        pl.xticks(fontsize=16)
        pl.yticks(fontsize=16)

        if xlim is not None and isinstance(xlim, list):
            pl.xlim(xlim)
        if ylim is not None and isinstance(ylim, list):
            pl.ylim(ylim)
        pl.savefig('raw_data_{0:d}.png'.format(n_data_type))
        pl.close()


def plot_route(map_components, route_ends, segments, path_indices):
    """
    Plot a route on top of a (already) plotted map

    Inputs:
        map_components is a pd.DataFrame with the raw NYC data
        route_ends is a list of 2 str with the form lon_lat each
        segments is a pd.DataFrame with the processed segments
        path indices is a list of the rows in segments that are part of the
            route
    """
    # plotting the map
    subplot_axes = plot_raw_data(map_components)  # ,
    #                              xlim=[-73.985, -73.955],
    #                              ylim=[40.760, 40.785])

    # plots starting and end point
    lon_start, lat_start = names.name_to_lon_lat(route_ends[0])
    pl.plot(lon_start, lat_start, 's', color='#ff33cc', markersize=12)
    lon_end, lat_end = names.name_to_lon_lat(route_ends[1])
    pl.plot(lon_end, lat_end, 'o', color='#ff33cc', markersize=12)

    # plotting the route
    if path_indices is not None:
        segments_gpd = gpd.GeoDataFrame(segments)
        route = segments_gpd.iloc[np.array(path_indices)]
        route.plot(ax=subplot_axes, color='k', linewidth=4)
    pl.savefig('path_run.png')

from flask import render_template
from flask import request
from flaskexample import app
from running_heaven.code.lib import data_handler
from running_heaven.code.lib import locations
from running_heaven.code import route_optimizer
from running_heaven.code.lib import google_map_api
import numpy as np


data_hand = data_handler.DataHandler()
segments = data_hand.load_processed_data()


@app.route('/')
@app.route('/input')
def input():
    return render_template("input.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/output')
def output():

    # object to interact with Google Map API
    gmaps = google_map_api.GoogleMapApi()

    # distance to run
    length = request.args.get('length')
    if length == '':
        length = '5.'
    length = float(length)

    # distance units
    if request.values['units'] == 'km':
        units = 'km'
    elif request.values['units'] == 'miles':
        units = 'miles'

    # starting point
    pt1_address = request.args.get('pt1')
    if pt1_address == '':
        pt1_address = 'Lexington Ave. and 62nd St.'
    pt1 = gmaps.get_lon_lat_from_address(pt1_address + ', Manhattan, NY')

    # end point
    if request.values.has_key('random'):
        pt2 = locations.select_random_point(segments, pt1, length)
        # pt2_address = gmaps.get_address_from_lon_lat(pt2)
        pt2_address = ''
    else:
        pt2_address = request.args.get('pt2')
        if pt2_address == '':
            pt2_address = 'Lexington Ave. and 63rd St.'
        pt2 = gmaps.get_lon_lat_from_address(pt2_address + ', Manhattan, NY')
    # combined points
    pt = (pt1, pt2)

    # sets the weight to 0 if a preference is not checked
    cost_weights = [np.nan] * 5
    for n_weight_type, weight_type in enumerate(['parks', 'trees',
                                                 'intersections']):
        if not request.values.has_key(weight_type):
            cost_weights[2+n_weight_type] = 0.

    # run the optimizer
    route_app = route_optimizer.RunRouteOptimizer(show=False)
    route_results = route_app.run(pt, length, units, cost_weights=cost_weights)
    d = route_results[0]
    path = route_results[1]

    # point to zoom on with Google Map
    center_lat = np.array(path)[:, 0].astype(float).mean()
    center_lon = np.array(path)[:, 1].astype(float).mean()

    # update the web-app
    return render_template("output.html",
                           length_requested='{0:.2f}'.format(length),
                           actual_length='{0:.2f}'.format(d),
                           route=path,
                           center_lat=center_lat,
                           center_lon=center_lon,
                           start_point=pt1_address,
                           end_point=pt2_address,
                           length=length,
                           units=units,
                           api_key=gmaps.gmap_key)

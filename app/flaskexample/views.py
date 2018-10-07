#!/usr/bin/env
"""
This code interacts between html pages and the python backend with flask
"""
from flask import render_template
from flask import request
import numpy as np
from flaskexample import app
from running_heaven.code.lib import data_handler
from running_heaven.code.lib import locations
from running_heaven.code import route_optimizer
from running_heaven.code.lib import google_map_api

# loading the datra only once
DATA_HAND = data_handler.DataHandler()
SEGMENTS = DATA_HAND.load_processed_data()
# object to interact with Google Map API
GMAPS = google_map_api.GoogleMapApi()

DEFAULT_VALUES = {'start': 'Lexington Ave. and 62nd St.',
                  'end': 'Lexington Ave. and 63rd St.',
                  'run_length': '5.'}

@app.route('/')
@app.route('/input')
def input_page():
    """
    Main page leads to the input webpage
    """
    return render_template("input.html")


@app.route('/about')
def about():
    """
    About webpage
    """
    return render_template("about.html")

def extract_value(field_name, default_value):
    """
    Given a field_name, extracts the value from teh html page, or sets it to
    the default value

    Inputs:
        field_name is a str and is defined in the html page
        default_value is the value to set the output to when the field is empty
            and is a str

    Output:
        variable is a str
    """
    variable = request.args.get(field_name)
    if variable == '':
        variable = default_value
    return variable


def get_address_and_point(field_name, default_value):
    """
    get the address and 'longitude_latitide' from the point defined in the html page

    Inputs:
        field_name is the variable name in the html page (str)
        default_value is a str and is the default address if the field is empty

    Outputs:
        adress is the address corresponding to the variable field_name
        point is a string 'longitude_latitude'
    """
    address = extract_value(field_name, default_value)
    full_address = address + ', Manhattan, NY'
    point = GMAPS.get_lon_lat_from_address(full_address)
    return address, point


def get_weights(request_values):
    """
    define the weights for the cost function

    Input:
        request_values is the output of request.values

    Output:
        list of weights
    """
    cost_weights = [np.nan] * 5
    for n_weight_type, weight_type in enumerate(['parks', 'trees',
                                                 'intersections']):
        if weight_type not in request_values:
            cost_weights[2+n_weight_type] = 0.
    return cost_weights


@app.route('/output')
def output_page():
    """
    Output of the code webpage
    """

    # distance to run
    run_length = float(extract_value('length', DEFAULT_VALUES['run_length']))

    # distance units
    if request.values['units'] == 'km':
        units = 'km'
    elif request.values['units'] == 'miles':
        units = 'miles'

    # starting point
    start_address, start_point = get_address_and_point('pt1',
                                                       DEFAULT_VALUES['start'])

    # end point
    if 'random' in request.values:
        end_point = locations.select_random_point(SEGMENTS, start_point,
                                                  run_length, units)
        end_address = ''
    else:
        end_address, end_point = get_address_and_point('pt2',
                                                       DEFAULT_VALUES['end'])

    # combined points
    route_ends = (start_point, end_point)

    # sets the weight to 0 if a preference is not checked
    cost_weights = get_weights(request.values)

    # run the optimizer
    route_app = route_optimizer.RunRouteOptimizer(num_points_per_segment=8,
                                                  show=False)
    route_results = route_app.run(route_ends, run_length, units,
                                  cost_weights=cost_weights)

    # point to zoom on with Google Map
    center_latitude = np.array(route_results[1])[:, 0].astype(float).mean()
    center_longitude = np.array(route_results[1])[:, 1].astype(float).mean()

    # update the web-app
    return render_template("output.html",
                           length_requested='{0:.2f}'.format(run_length),
                           actual_length='{0:.2f}'.format(route_results[0]),
                           route=route_results[1],
                           center_lat=center_latitude,
                           center_lon=center_longitude,
                           start_point=start_address,
                           end_point=end_address,
                           length=run_length,
                           units=units,
                           api_key=GMAPS.gmap_key)

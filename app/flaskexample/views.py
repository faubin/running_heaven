from flask import render_template
from flask import request
from flaskexample import app
import running_heaven.code.route_optimizer as route_optimizer
import running_heaven.code.google_map_api as google_map_api
import numpy as np


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

    # starting point
    pt1_address = request.args.get('pt1')
    if pt1_address == '':
        pt1_address = 'Lexington Ave. and 62nd St.'
    pt1 = gmaps.get_lon_lat_from_address(pt1_address + ', Manhattan, NY')
    # end point
    pt2_address = request.args.get('pt2')
    if pt2_address == '':
        pt2_address = 'Lexington Ave. and 63rd St.'
    pt2 = gmaps.get_lon_lat_from_address(pt2_address + ', Manhattan, NY')
    # combined points
    pt = (pt1, pt2)

    # distance to run
    length = request.args.get('length')
    if length == '':
        length = '5.'
    length = float(length)

    # sets the weight to 0 if a preference is not checked
    cost_weights = [np.nan] * 5
    for n_weight_type, weight_type in enumerate(['parks', 'trees',
                                                 'intersections']):
        if not request.values.has_key(weight_type):
            cost_weights[2+n_weight_type] = 0.

    # run the optimizer
    route_app = route_optimizer.RunRouteOptimizer(show=False)
    d, path = route_app.run(pt, length, cost_weights=cost_weights)

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
                           api_key=gmaps.gmap_key)

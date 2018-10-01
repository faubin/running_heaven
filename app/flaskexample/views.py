from flask import render_template
from flask import request
from flaskexample import app
import running_heaven.code.route_optimizer as route_optimizer
import running_heaven
import os
import googlemaps


if '_path' in dir(running_heaven.__path__):
    running_heaven_path = running_heaven.__path__._path[0]
else:
    running_heaven_path = running_heaven.__path__[0]


def get_lon_lat_from_address(gmaps, address):
    """
    """
    geocode_result = gmaps.geocode(address)
    lon_lat = (geocode_result[0]['geometry']['location']['lng'],
               geocode_result[0]['geometry']['location']['lat'])
    return lon_lat

@app.route('/')
@app.route('/input')
def input():
    return render_template("input.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/output')
def output():

    # load the Google API key for Google Map
    f = open(os.path.join(running_heaven_path, 'app/keys/googleApiKey.txt'),
             'r')
    api_key = f.readline()
    f.close()

    # google map object
    gmaps = googlemaps.Client(key=api_key)


    # starting point
    pt1_address = request.args.get('pt1')
    pt1 = get_lon_lat_from_address(gmaps, pt1_address + ', Manhattan, NY')
    pt1 = '{0:f}_{1:f}'.format(*pt1)
    # end point
    pt2_address = request.args.get('pt2')
    pt2 = get_lon_lat_from_address(gmaps, pt2_address + ', Manhattan, NY')
    pt2 = '{0:f}_{1:f}'.format(*pt2)
    # combined points
    pt = (pt1, pt2)
    # distance to run
    length = float(request.args.get('length'))

    # run the optimizer
    app = route_optimizer.RunRouteOptimizer(show=False)
    d, path = app.run(pt, length)

    # update the app
    return render_template("output.html",
                           length_requested='{0:.2f}'.format(length),
                           actual_length='{0:.2f}'.format(d),
                           x=path,
                           start_point=pt1_address,
                           end_point=pt2_address, length=length,
                           api_key=api_key)


from flask import render_template
from flask import request
from flaskexample import app
import running_heaven.code.route_optimizer as route_optimizer
import running_heaven.code.google_map_api as google_map_api


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
    pt1 = gmaps.get_lon_lat_from_address(pt1_address + ', Manhattan, NY')
    pt1 = '{0:f}_{1:f}'.format(*pt1)
    # end point
    pt2_address = request.args.get('pt2')
    pt2 = gmaps.get_lon_lat_from_address(pt2_address + ', Manhattan, NY')
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
                           api_key=gmaps.gmap_key)


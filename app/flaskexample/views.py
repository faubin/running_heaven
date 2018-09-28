from flask import render_template
from flask import request
from flaskexample import app
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
#import pandas as pd
#import psycopg2
#from flaskexample.a_Model import ModelIt
import running_heaven.code.route_optimizer as route_optimizer
import running_heaven
import os


if '_path' in dir(running_heaven.__path__):
    running_heaven_path = running_heaven.__path__._path[0]
else:
    running_heaven_path = running_heaven.__path__[0]


@app.route('/')
@app.route('/input')
def input():
    return render_template("input.html")


@app.route('/output')
def output():
    pt1 = request.args.get('pt1')
    pt2 = request.args.get('pt2')
    pt = (pt1, pt2)
    length = float(request.args.get('length'))
    app = route_optimizer.RunRouteOptimizer(show=False)
    d, path = app.run(pt, length)
    # return render_template("output.html", length_requested=str(length),

    f = open(os.path.join(running_heaven_path, 'app/keys/googleApiKey.txt'),
             'r')
    api_key = f.readline()
    f.close()
    print(api_key)

    #toto = [['40.763480', '-73.967271'],
    #        ['40.773480', '-73.967271'],
    #        ['40.763480', '-73.977271']]
    print(path)
    return render_template("output.html",
                           length_requested='{0:.2f}'.format(length),
                           actual_length='{0:.2f}'.format(d),
#                           #path_lat_lon=toto,
                           x=path,
                           start_point=pt1,
                           end_point=pt2, length=length,
                           api_key=api_key)

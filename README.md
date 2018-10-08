# Running Heaven: Optimize for the Most Pleasant Running Route

Running Heaven is a web-app I built in 3 weeks as an Insight Data Science Fellow in September 2018.

It uses data from the NYC Open Data website:
  * parks (https://data.cityofnewyork.us/City-Government/Parks-Zones/rjaj-zgq7)
  * streets (https://data.cityofnewyork.us/City-Government/NYC-Street-Centerline-CSCL-/exjm-f27b)
  * pedestrian roads (https://data.cityofnewyork.us/dataset/Sidewalk-Centerline/a9xv-vek9)
  * trees (https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/pi5s-9p35)

I implemented a modified Dijkstra's algorithm to optimize running routes bases on the pleasantness of the route given a constraint on distance. The alogrithm favors high tree density and pedestrian routes and minimizes street intersections.

The web-app lives at http://runningheaven.space/

The code currently only supports Manhattan. It will soon support NYC.

Below is an example of a 5 km route through Central Park. Parks (yellow), streets (red), pedestrian roads (blue), and trees (green) are shown with the optimized running route (black) from the starting point (quare magenta) to the end point (circle magents)

<p align='center'>
<img src='figures/example_route.png' width='650'>
</p>


# To Run Locally
  * Download the data (a script will be provided in the near future, but download the data manually from the links below and move the files to running_heaven/raw_data/).
  * Run build_database.py to process the raw data
  * Run route_optimization.py to optimize routes. The arguments are hardcoded for now, but it will be possible to enter them from the command line soon.

# Packages to install
  * python 3 (tested with python 3.6.5)
  * numpy
  * scipy
  * pandas
  * matplotlib
  * flask
  * scikit-learn
  * jupyter
  * seaborn
  * geopandas
  * googlemaps
  * requests
  * flask (only for web-app)
  * gunicorn (only for web-app server)


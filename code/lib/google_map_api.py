#!/usr/bin/env
"""
This class is an interface to communicate with the Google Map API
"""
import googlemaps
from running_heaven.code.lib import api_keys
# from running_heaven.code.lib import names


class GoogleMapApi(api_keys.ApiKeys):
    """
    A class to interface with the Google Map API
    """
    def __init__(self):
        """
        """
        api_keys.ApiKeys.__init__(self)

        # google map object
        self.gmaps = googlemaps.Client(key=self.gmap_key)
        return

    def get_lon_lat_from_address(self, address):
        """
        Given an address, returns the longitude and latitude ('lon_lat') in
        degrees
        """
        geocode_result = self.gmaps.geocode(address)
        lon_lat = (geocode_result[0]['geometry']['location']['lng'],
                   geocode_result[0]['geometry']['location']['lat'])
        lon_lat_str = '{0:f}_{1:f}'.format(*lon_lat)
        return lon_lat_str

#    def get_address_from_lon_lat(self, lon_lat_str):
#        """
#        Given an address, returns the longitude and latitude ('lon_lat') in
#        degrees
#
#        !!!
#        Does not quite works. Get a list of results and it is not obvious how
#        to get an address out!!!
#        !!!
#        """
#        lon, lat = names.name_to_lon_lat(lon_lat_str)
#        geocode_result = self.gmaps.reverse_geocode((lat, lon))
#        for i in range(len(geocode_result)):
#            print(geocode_result[i]['address_components'][0]['short_name'])
        # lon_lat = (geocode_result[0]['geometry']['location']['lng'],
        #            geocode_result[0]['geometry']['location']['lat'])
        # lon_lat_str = '{0:f}_{1:f}'.format(*lon_lat)
        # return lon_lat_str

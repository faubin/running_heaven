import googlemaps
import os
from running_heaven.code import api_keys


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
        return lon_lat

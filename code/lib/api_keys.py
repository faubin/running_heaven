#!/usr/bin/env
"""
This class loads api keys
"""
import os
from running_heaven.code.lib import core


class ApiKeys(core.HeavenCore):
    """
    A class to interface with the Google Map API
    """
    def __init__(self):
        core.HeavenCore.__init__(self)
        self.gmap_key = self.load_google_api_key()
        return

    def load_google_api_key(self, key_file_name='googleApiKey.txt'):
        """
        Loads the Google Map API key
        """
        path = os.path.join(self.running_heaven_path, 'app', 'keys')

        # error checking
        if key_file_name not in os.listdir(path):
            error_text = 'I expect a file {0:s} '.format(key_file_name)
            error_text += 'in {0:s} to use the Google Map API.'.format(path)
            raise ValueError(error_text)

        # load the API key
        full_key_file_name = os.path.join(path, key_file_name)
        with open(full_key_file_name, 'r') as key_file:
            api_key = key_file.readline()

        return api_key

if __name__ == "__main__":
    APP = ApiKeys()
    print(APP.running_heaven_path)

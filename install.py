#!/usr/bin/env
"""
This code install the required libraries and downloads the data for the library
It assumes the user runs the code in a conda virtual environment.
"""
import os
import sys
import requests
from running_heaven.code.lib import core


class RunningHeavenInstaller(core.HeavenCore):
    """
    A class to setup the library
    """
    def __init__(self):
        core.HeavenCore.__init__(self)
        self.libraries = ['numpy',
                          'scipy',
                          'pandas',
                          'matplotlib',
                          'flask',
                          'scilit-learn',
                          'jupyter',
                          'seaborn',
                          'geopandas',
                          'googlemaps',
                          'requests']

        self.nyc_open_data_url = os.path.join('https://',
                                              'data.cityofnewyork.us',
                                              'api',
                                              'geospatial')
        self.data_id = {'Parks Zones': 'rjaj-zgq7',
                        'Borough Boundaries': 'tqmj-j8zm',
                        'NYC Street Centerline (CSCL)': 'exjm-f27b',
                        'Sidewalk Centerline': 'a9xv-vek9'}
        self.api_parameters = {'method': 'export', 'format': 'GeoJson'}

        self.path_for_bashrc = '/'.join(os.getcwd().split('/')[:-1])


    def install_packages(self):
        """
        Install the required packages in a conda virtual environment.
        """
        for package in self.libraries:
            print('conda install {0:s}'.format(package))
            os.system('conda install {0:s}'.format(package))

    def download_data(self):
        """
        Downloads the data from NYC open data
        """
        # make sure the raw_data folder exists
        raw_data_path = os.path.join(self.running_heaven_path, 'raw_data')
        if 'raw_data' not in os.listdir(self.running_heaven_path):
            os.mkdir(raw_data_path)

        # downloading the data
        for file_name in self.data_id:
            print('downloading {0:s}'.format(file_name))
            url = os.path.join(self.nyc_open_data_url,
                               self.data_id[file_name])

            request = requests.get(url, self.api_parameters)
            save_file_name = os.path.join(raw_data_path,
                                          '{0:s}.geojson'.format(file_name))
            with open(save_file_name, 'wb') as file_:
                file_.write(request.content)


if __name__ == "__main__":
    APP = RunningHeavenInstaller()

    print('\nMake sure the following 2 lines are in your .bashrc file:')
    print('PYTHONPATH=$PYTHONPATH:{0:s}'.format(APP.path_for_bashrc))
    print('export PYTHONPATH')
    print('And run "source ~/.bashrc" after modifying the file')
    input('Press enter when you are done')

    INSTALL_PACKAGES = False
    DOWNLOAD_DATA = False
    if len(sys.argv) == 1:
        INSTALL_PACKAGES = True
        DOWNLOAD_DATA = True
    elif 'install' in sys.argv:
        INSTALL_PACKAGES = True
    elif 'download' in sys.argv:
        DOWNLOAD_DATA = True

    if INSTALL_PACKAGES:
        APP.install_packages()
    if DOWNLOAD_DATA:
        APP.download_data()

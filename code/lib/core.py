#!/usr/bin/env
"""
This is a class to be inherited which loads variables common to other classes
"""
import running_heaven


class HeavenCore():
    """
    A class to be inherited by the running heaven code
    """
    def __init__(self):
        if hasattr(running_heaven.__path__, '_path'):
            self.running_heaven_path = running_heaven.__path__._path[0]
        else:
            self.running_heaven_path = running_heaven.__path__[0]
        return

if __name__ == "__main__":
    APP = HeavenCore()

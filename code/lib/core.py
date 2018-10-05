import running_heaven
import os


class HeavenCore():
    """
    A class to be inherited by the running heaven code
    """
    def __init__(self):
        if '_path' in dir(running_heaven.__path__):
            self.running_heaven_path = running_heaven.__path__._path[0]
        else:
            self.running_heaven_path = running_heaven.__path__[0]
        return

if __name__ == "__main__":
    print('Implement test here')

"""
PathHandler module to be able to change the path to the internal dataset
"""
import os


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MyPathHandler(object, metaclass=Singleton):
    """
    Path Handler for tests
    """

    def __init__(self,
                 path_data=None):
        """
        initialise path to path pointed to by environment variable "DATA"
        """
        if path_data is None:
            path_data = os.path.join(os.environ["DATA"], "internal_data/gt/")
        self.path_data = path_data

    def set_path(self, path_data):
        """
        set path
        """
        self.path_data = path_data

    def get_path(self):
        """
        get path
        """
        return self.path_data


class PathOverWrite(object):
    """
    "with" facility for tests with differents paths to data
    """
    def __init__(self, path_data):
        self.new_path = path_data
        self.tmp_path = None

    def __enter__(self):
        """
        when entering "with", change path
        """
        self.tmp_path = MyPathHandler().get_path()
        MyPathHandler().set_path(self.new_path)
        return self

    def __exit__(self, type, value, traceback):
        """
        when exiting "with", reset path to previous value
        """
        MyPathHandler().set_path(self.tmp_path)

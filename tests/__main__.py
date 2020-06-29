import unittest
import logging
logging.basicConfig(level=logging.INFO)

from .test_basic import *
from .test_autocomplete import AutocompleteTests


if __name__ == '__main__':
    # print(dir())
    unittest.main()

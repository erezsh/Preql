import sys
import unittest
import logging
logging.basicConfig(level=logging.INFO)

from .test_basic import *
from .test_autocomplete import AutocompleteTests

minimal = [
    AutocompleteTests,
    TestTypes,
    TestFlow,
    TestFunctions,
    BasicTests_0_Normal_Lt,
]

TESTS_SUITES = {
    'minimal': minimal
}



if __name__ == '__main__':
    try:
        tests = TESTS_SUITES[sys.argv[1]]
    except KeyError:
        unittest.main()
    else:
        suite = unittest.TestSuite()
        for t in tests:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(t))
        unittest.TextTestRunner().run(suite)


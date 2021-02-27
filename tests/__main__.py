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

def run_test_suite(suit):
    tests = TESTS_SUITES[suit]
    suite = unittest.TestSuite()
    for t in tests:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(t))
    unittest.TextTestRunner().run(suite)

if __name__ == '__main__':
    try:
        tests = run_test_suite(sys.argv[1])
    except LookupError:
        unittest.main()
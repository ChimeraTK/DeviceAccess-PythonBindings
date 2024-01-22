#!/usr/bin/env python3

import unittest
import numpy
import sys
import os

# fmt: off
# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module. Formatting is switched off,
# so the import is not sorted into the others.
sys.path.insert(0, os.path.abspath(os.curdir))
import testmodule
# fmt: on


class TestHelpers(unittest.TestCase):

    def testExtractDataType(self):
        a = numpy.empty([1], dtype=numpy.int32)
        self.assertEqual(testmodule.extractDataType(a),
                         testmodule.numpyDataTypes.INT32)

        a = numpy.empty([1, 2], dtype=numpy.int32)
        self.assertEqual(testmodule.extractDataType(a),
                         testmodule.numpyDataTypes.INT32)

        a = numpy.empty([1, 2], dtype=numpy.int64)
        self.assertEqual(testmodule.extractDataType(a),
                         testmodule.numpyDataTypes.INT64)

        a = numpy.empty([1, 2], dtype=numpy.float32)
        self.assertEqual(testmodule.extractDataType(a),
                         testmodule.numpyDataTypes.FLOAT32)

        a = numpy.empty([1, 2], dtype=numpy.float64)
        self.assertEqual(testmodule.extractDataType(a),
                         testmodule.numpyDataTypes.FLOAT64)

        a = numpy.empty([1, 2], dtype=numpy.bool_)
        self.assertEqual(testmodule.extractDataType(a),
                         testmodule.numpyDataTypes.USUPPORTED_TYPE)

    def testNumpyObjManager(self):
        a = numpy.empty([1, 2], dtype=numpy.float64)
        try:
            testmodule.testNumpyObjManager(a)
        except BaseException:
            self.fail("exception in testNumpyObjManager")


if __name__ == '__main__':
    unittest.main()

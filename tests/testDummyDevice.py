#!/usr/bin/env python3
import numpy
import unittest
import os
import sys

# fmt: off
# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module. Formatting is switched off,
# so the import is not sorted into the others.
sys.path.insert(0, os.path.abspath(os.curdir))
import mtca4u
# fmt: on


class TestDummyDevice(unittest.TestCase):

    def testCreateDummyDevice(self):
        self.assertRaisesRegexp(RuntimeError, "Cannot open file \".*someBogusMapFile.map\"", mtca4u.Device,
                                "sdm://./dummy=someBogusMapFile.map")


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
import _da_python_bindings as mtca4udeviceaccess
import os
import sys
import unittest
import numpy

# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module.
sys.path.insert(0, os.path.abspath(os.curdir))


class TestOpeningException(unittest.TestCase):

    def testExceptionWhenOpening(self):
        # We use the dummy with a bad map file as we know that deviceaccess
        # is throwing here.
        self.assertRaisesRegexp(RuntimeError, "Cannot open file \".*badMapFile.map\"",
                                mtca4udeviceaccess.createDevice, "sdm://./dummy=badMapFile.map")


if __name__ == '__main__':
    unittest.main()

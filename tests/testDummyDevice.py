#!/usr/bin/env python3
import os
import sys

# This is a hack so mtca4u is found in the current dir.
sys.path.insert(0, os.path.abspath(os.curdir))

import mtca4u
import unittest
import numpy


class TestDummyDevice(unittest.TestCase):

    def testCreateDummyDevice(self):
        self.assertRaisesRegexp(RuntimeError, "Cannot open file \".*someBogusMapFile.map\"", mtca4u.Device,
                                "sdm://./dummy=someBogusMapFile.map")


if __name__ == '__main__':
    unittest.main()

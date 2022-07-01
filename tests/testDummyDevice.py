#!/usr/bin/env python3

import mtca4u
import os
import sys
import unittest
import numpy

# This is a hack for nw
sys.path.insert(0, os.path.abspath(os.curdir))


class TestDummyDevice(unittest.TestCase):

    def testCreateDummyDevice(self):
        self.assertRaisesRegexp(RuntimeError, "Cannot open file \".*someBogusMapFile.map\"", mtca4u.Device,
                                "sdm://./dummy=someBogusMapFile.map")


if __name__ == '__main__':
    unittest.main()

#! /usr/bin/python

import os
import sys
import  unittest
import numpy

# This is a hack for nw
sys.path.insert(0,os.path.abspath(os.curdir))
import mtca4udeviceaccess

class TestDummyDevice(unittest.TestCase):

    def testCreateDummyDevice(self):
        self.assertRaisesRegexp(RuntimeError, "Cannot open file \"someBogusMapFile.map\""
                , mtca4udeviceaccess.createDevice,
                "deviceInformation/mtcadummy.map", "someBogusMapFile.map")

if __name__ == '__main__':
    unittest.main()

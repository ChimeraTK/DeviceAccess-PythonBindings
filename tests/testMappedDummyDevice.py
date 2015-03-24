#! /usr/bin/python

import os
import sys
import  unittest
import numpy

# This is a hack for nw
sys.path.insert(0,os.path.abspath(os.curdir))
import mtcamappeddevice

class TestDummyDevice(unittest.TestCase):

    def testCreateDummyDevice(self):
        self.assertRaisesRegexp(RuntimeError, "Mapped Dummy Device expects"
                " first and second parameters to be the same map file"
                , mtcamappeddevice.createDevice,
                "mapfiles/mtcadummy.map", "someBogusMapFile.map")

if __name__ == '__main__':
    unittest.main()

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
        self.assertRaisesRegexp(RuntimeError, "Cannot open device: : No such "
                "file or directory", mtcamappeddevice.createDevice, "")
        self.assertRaisesRegexp(RuntimeError, "Cannot open device: "
                "some_non_existent_device: No such file or directory", mtcamappeddevice.createDevice,
                "some_non_existent_device")
        self.assertRaisesRegexp(RuntimeError, "Cannot open device: "
                "../mapfiles/mtcadummy.map1: No such file or directory", mtcamappeddevice.createDevice,
                "../mapfiles/mtcadummy.map1")

if __name__ == '__main__':
    unittest.main()

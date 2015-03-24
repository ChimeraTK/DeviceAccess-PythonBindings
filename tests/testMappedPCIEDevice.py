#! /usr/bin/python
import os
import sys
import  unittest
import numpy

# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the mtacamappeddevice module. 
sys.path.insert(0,os.path.abspath(os.curdir))
import mtcamappeddevice

class TestMappedPCIEDevice(unittest.TestCase):

    def testCreateMappedPCIEDevice(self):
        self.assertRaisesRegexp(RuntimeError, "Cannot open device: : No such "
                "file or directory", mtcamappeddevice.createDevice, "", "")

if __name__ == '__main__':
    unittest.main()

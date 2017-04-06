#!/usr/bin/env ${python_interpreter}
import os
import sys
import  unittest
import numpy

# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module. 
sys.path.insert(0,os.path.abspath(os.curdir))
import mtca4udeviceaccess

class TestPCIEDevice(unittest.TestCase):

    def testCreatePCIEDevice(self):
        self.assertRaisesRegexp(RuntimeError, "Cannot open file \"\"",
                                 mtca4udeviceaccess.createDevice, "", "")

if __name__ == '__main__':
    unittest.main()

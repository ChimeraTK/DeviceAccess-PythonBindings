#! /usr/bin/python
import os
import sys
import  unittest
import numpy

# This is a hack for nw
sys.path.insert(0,os.path.abspath(os.curdir))
import mtca4ucore

class TestPCIEDevice(unittest.TestCase):

    def testCreatePCIEDevice(self):
        pass
#         self.assertRaisesRegexp(RuntimeError, "Functionality not available yet"
#                 , mtca4ucore.createDevice, "")
#         self.assertRaisesRegexp(RuntimeError, "Functionality not available yet"
#                  , mtca4ucore.createDevice,
#                 "some_non_existent_device")

if __name__ == '__main__':
    unittest.main()

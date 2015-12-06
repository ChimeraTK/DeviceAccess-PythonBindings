#! /usr/bin/python

import os
import sys
import  unittest
import numpy

# This is a hack for nw
sys.path.insert(0,os.path.abspath(os.curdir))
import mtca4ucore

class TestDummyDevice(unittest.TestCase):

    def testCreateDummyDevice(self):
        pass
        # commenting out these tests as they are no longer applicable. We do not
        # have a version of create device that takes in just the card alias name
        # yet
#         self.assertRaisesRegexp(RuntimeError, "Functionality not available yet"
#                 , mtca4ucore.createDevice, "")
#         self.assertRaisesRegexp(RuntimeError, "Functionality not available yet"
#                 , mtca4ucore.createDevice,
#                 "some_non_existent_device")
#         self.assertRaisesRegexp(RuntimeError, "Functionality not available yet"
#                 , mtca4ucore.createDevice,
#                 "../deviceInformation/mtcadummy.map1")

if __name__ == '__main__':
    unittest.main()

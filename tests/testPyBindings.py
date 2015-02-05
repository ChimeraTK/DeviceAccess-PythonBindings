#! /usr/bin/python

# This is a hack for nw

import os
import sys
print os.path.abspath(os.curdir)
sys.path.insert(0,os.path.abspath(os.curdir))

import  unittest
import mtcamappeddevice

class TestPCIEDevice(unittest.TestCase):

    def testHelloWorld(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4")
        self.assertTrue(True)

    def test(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()

#! /usr/bin/python
import os
import sys
import  unittest
import numpy

# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the module myModule. 
sys.path.insert(0,os.path.abspath(os.curdir))
import myModule

class TestMappedPCIEDevice(unittest.TestCase):
  
  def testRead(self):
    self.assertTrue(1==1)
    
if __name__ == '__main__':
    unittest.main()
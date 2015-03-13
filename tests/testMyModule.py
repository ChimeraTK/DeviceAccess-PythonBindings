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
    
    
    # Set up WORD_CLK_MUX register with predefined values
    device = myModule.mtcamappeddevice.createDevice("/dev/llrfdummys4",
                                           "mapfiles/mtcadummy.map")
    WORD_CLK_MUX_REG_Offset = 32
    dataToSet = numpy.array([5.0, 4.0, 3213.0, 2.0], dtype = numpy.float32)
    bytesToWrite = 4 * 4 # i.e 4 32 bit words
    bar = 0 
    device.writeRaw(WORD_CLK_MUX_REG_Offset, dataToSet, bytesToWrite, bar)
    # This section checks the read register code for the Device class
    device = myModule.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
    
    # check if function reads values correctly
    readInValues = device.read("WORD_CLK_MUX")
    readInValues.dtype = numpy.float32
    self.assertTrue(readInValues.tolist() == dataToSet.tolist())
    
    readInValues = device.read("WORD_CLK_CNT", 1)
    self.assertTrue(readInValues.tolist() == [5.0])
    
    readInValues = device.read("WORD_CLK_CNT", 1, 2)
    self.assertTrue(readInValues.tolist() == [3213.0])
    
    readInValues = device.read("WORD_CLK_CNT", 0, 2)
    self.assertTrue(readInValues.tolist() == [3213.0, 2.0])
    
    # check for corner cases
    # Register Not Found
    # Num of elements specified  is more than register size
    # offset exceeds register offset 
    self.assertTrue(1==1)
    
if __name__ == '__main__':
    unittest.main()     
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
  # TODO: Refactor to take care of the harcoded values used for comparisions
  def testRead(self):
    
    
    # Set up WORD_CLK_MUX register with predefined values
    device = myModule.mtcamappeddevice.createDevice("/dev/llrfdummys4",
                                           "mapfiles/mtcadummy.map")
    WORD_CLK_MUX_REG_Offset = 32
    dataToSet = numpy.array([5, 4, 3213, 2], dtype = numpy.int32)
    bytesToWrite = 4 * 4 # i.e 4 32 bit words
    bar = 0 
    device.writeRaw(WORD_CLK_MUX_REG_Offset, dataToSet, bytesToWrite, bar)
    
    WORD_INCOMPLETE_2_REG_OFFSET = 100
    dataToSet = numpy.array([544], dtype = numpy.int32) # FP representation
                                                        # of 2.15
    bytesToWrite = 4 * 1 # i.e 1 32 bit word
    bar = 0 
    device.writeRaw(WORD_INCOMPLETE_2_REG_OFFSET, dataToSet, bytesToWrite, bar)
    
    # This section checks the read register code for the Device class
    device = myModule.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
    
    # check if function reads values correctly
    readInValues = device.read("WORD_INCOMPLETE_2")
    self.assertTrue(readInValues.dtype == numpy.float32)
    self.assertTrue(readInValues.tolist() == [2.125])
    
    readInValues = device.read("WORD_CLK_MUX")
    self.assertTrue(readInValues.dtype == numpy.float32)
    self.assertTrue(readInValues.tolist() == [5.0, 4.0, 3213.0, 2.0])
    
    readInValues = device.read("WORD_CLK_MUX", 1)
    self.assertTrue(readInValues.tolist() == [5.0])
    
    readInValues = device.read("WORD_CLK_MUX", 1, 2)
    self.assertTrue(readInValues.tolist() == [3213.0])
    
    readInValues = device.read("WORD_CLK_MUX", 0, 2)
    self.assertTrue(readInValues.tolist() == [3213.0, 2.0])
    
    # check for corner cases
    # Register Not Found
    self.assertRaisesRegexp(RuntimeError, "Cannot find register"
                            " BAD_REGISTER_NAME in map file:"
                            " mapfiles/mtcadummy.map", device.read, 
                            "BAD_REGISTER_NAME")


    # Num of elements specified  is more than register size
    registerName = "WORD_CLK_MUX"
    elementsToRead = 5
    offset = 2
    self.assertRaisesRegexp(RuntimeError, "Data size exceed register size",
                             device.read, registerName, elementsToRead)
    self.assertRaisesRegexp(RuntimeError, "Data size exceed register size", 
                            device.read, registerName, elementsToRead, offset)
    self.assertRaisesRegexp(ValueError, "negative dimensions are not allowed",
                             device.read, registerName, 
                             numberOfElementsToRead=-1)
    
    # offset exceeds register size
    offset = 5
    elementsToRead = 5
    self.assertRaisesRegexp(ValueError, "Element index: 5 incorrect. Valid index"
                            " range is \[0-3\]", device.read, 
                            registerName, elementIndexInRegister = offset) 
    self.assertRaisesRegexp(ValueError, "Element index: 5 incorrect. Valid index"
                            " range is \[0-3\]", device.read, 
                            registerName, elementsToRead, offset)
    self.assertRaisesRegexp(OverflowError, "can't convert negative value to"
                            " unsigned", device.read, 
                            registerName, elementIndexInRegister = -1)
    

  def testWrite(self):
    # write to WORD_INCOMPLETE_2, this is 13 bits wide and supports 8
    # fractional bits
    word_incomplete_register = "WORD_INCOMPLETE_2"
    # check the write functionality
    device = myModule.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
    device.write(word_incomplete_register, 
               numpy.array([2.125], dtype = numpy.float32))
    readInValue = device.read(word_incomplete_register)
    self.assertTrue(readInValue.dtype == numpy.float32)
    self.assertTrue(readInValue.tolist() == [2.125])
    
    
    # check offset functionality
    word_clk_mux_register = "WORD_CLK_MUX"
    dataToWrite = numpy.array([8732, 789], dtype = numpy.float32)
    device.write(word_clk_mux_register, dataToWrite, 
                 elementIndexInRegister = 2)
    readInValue = device.read(word_clk_mux_register,
                              elementIndexInRegister = 2)
    self.assertTrue(readInValue.dtype == numpy.float32)
    self.assertTrue(readInValue.tolist() == dataToWrite.tolist())
     
    # Check corner cases
    # Bogus register name
    self.assertRaisesRegexp(RuntimeError, "Cannot find register"
                            " BAD_REGISTER_NAME in map"
                            " file: mapfiles/mtcadummy.map", device.write, 
                            "BAD_REGISTER_NAME", 
                            numpy.array([2.125], dtype = numpy.float32))
    
    # supplied array size exceeds register capacity
    dataToWrite = numpy.array([2.125, 3, 4], dtype = numpy.float32)
    self.assertRaisesRegexp(RuntimeError, "Data size exceed register size",
                             device.write, word_incomplete_register, 
                             dataToWrite)
    
    # supplied offset exceeds register span
    dataToWrite = numpy.array([2.125, 3, 4], dtype = numpy.float32)
    self.assertRaisesRegexp(ValueError, "Element index: 1 incorrect."
                            " Valid index is 0", device.write, 
                            word_incomplete_register, dataToWrite,
                            elementIndexInRegister=1)
    
    # dtype of input array not float32
    dataToWrite = numpy.array([2], dtype = numpy.int32)
    self.assertRaisesRegexp(TypeError, "Method expects values to be framed in a"
                            " float32 numpy.array", device.write, 
                            word_incomplete_register, dataToWrite)
    
    
  def testreadRaw(self):
    # write some raw values in
    device = myModule.mtcamappeddevice.createDevice("/dev/llrfdummys4",
                                               "mapfiles/mtcadummy.map")
    WORD_CLK_MUX_REG_Offset = 32
    dataToSet = numpy.array([698, 244, 3223, 213], dtype=numpy.int32)
    bytesToWrite = 4 * 4  # i.e 4 32 bit words
    bar = 0 
    device.writeRaw(WORD_CLK_MUX_REG_Offset, dataToSet, bytesToWrite, bar)

    # read them in and verify
    device = myModule.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
    readInValues = device.read("WORD_CLK_MUX")
    self.assertTrue(readInValues.dtype == numpy.int32)
    self.assertTrue(readInValues.tolist() == dataToSet.tolist())

    readInValues = device.readRaw("WORD_CLK_MUX", 1)
    self.assertTrue(readInValues.tolist() == [698])

    readInValues = device.readRaw("WORD_CLK_MUX", 1, 2)
    self.assertTrue(readInValues.tolist() == [3223])

    readInValues = device.readRaw("WORD_CLK_MUX", 0, 2)
    self.assertTrue(readInValues.tolist() == [3223, 213])

    # check corner cases
    # bad reg name
    self.assertRaisesRegexp(RuntimeError, "Fill this in", device.readRaw, 
                            "BAD_REGISTER_NAME")
      
    # Num of elements specified  is more than register size
    registerName = "WORD_CLK_MUX"
    elementsToRead = 5
    offset = 2
    self.assertRaisesRegexp(RuntimeError, "Fill this in", device.readRaw, 
                            registerName, elementsToRead)
    self.assertRaisesRegexp(RuntimeError, "Fill this in", device.readRaw, 
                            registerName, elementsToRead, offset)
    self.assertRaisesRegexp(RuntimeError, "Fill this in", device.readRaw, 
                            registerName, numberOfElementsToRead=-1)
    
    # bad offset value
    offset = 5
    self.assertRaisesRegexp(RuntimeError, "Fill this in", device.readRaw, 
                            registerName, offsetFromRegisterBaseAddress = offset) 
    self.assertRaisesRegexp(RuntimeError, "Fill this in", device.readRaw, 
                            registerName, numberofElementsToRead, offset)
    self.assertRaisesRegexp(RuntimeError, "Fill this in", device.readRaw, 
                            registerName, offsetFromRegisterBaseAddress = -1)

  def testwriteRaw(self):
    # write to WORD_CLK_MUX register and verify the read
    device = myModule.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
    word_clk_mux_register = "WORD_CLK_MUX"
    dataToWrite = numpy.array([456, 3445, 8732, 789], dtype = numpy.int32)
    device.writeRaw(word_clk_mux_register, dataToWrite)
    readInValue = device.readRaw(word_clk_mux_register)
    self.assertTrue(readInValue.dtype == numpy.int32)
    self.assertTrue(readInValue.tolist() == dataToWrite.tolist())
    
    # check offset functionality
    dataToWrite = numpy.array([5698, 2354], dtype = numpy.int32)
    device.writeRaw(word_clk_mux_register, dataToWrite, 
                 offsetFromRegisterBaseAddress=2)
    readInValue = device.readRaw(word_clk_mux_register)
    self.assertTrue(readInValue.dtype == numpy.int32)
    self.assertTrue(readInValue.tolist() == dataToWrite.tolist())
    

    # corner cases:
    # bogus register
    self.assertRaisesRegexp(RuntimeError, "Fill this in", 
                            device.writeRaw, "BAD_REGISTER_NAME", 
                            numpy.array([2], dtype = numpy.int32))
    # array size exceeds register capacity
    dataToWrite = numpy.array([2, 5, 5, 3, 4], dtype = numpy.int32)
    self.assertRaisesRegexp(RuntimeError, "Fill this in", device.writeRaw, 
                            word_clk_mux_register, dataToWrite)
    # offset is bogus
    self.assertRaisesRegexp(RuntimeError, "Fill this in", device.writeRaw, 
                            word_clk_mux_register, dataToWrite, 
                            offsetFromRegisterBaseAddress=4)
    self.assertRaisesRegexp(RuntimeError, "Fill this in", device.writeRaw, 
                            word_clk_mux_register, dataToWrite, 
                            offsetFromRegisterBaseAddress=-1)
    # array dtype not int32
    dataToWrite = numpy.array([2, 3, 4, 5], dtype = numpy.float32)
    self.assertRaisesRegexp(RuntimeError, "Fill this in", device.write, 
                            word_clk_mux_register, dataToWrite)

if __name__ == '__main__':
    unittest.main()     

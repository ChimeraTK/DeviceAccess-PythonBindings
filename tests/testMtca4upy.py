#! /usr/bin/python
import os
import sys
import  unittest
import numpy

# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the module mtca4u. 
sys.path.insert(0,os.path.abspath(os.curdir))
import mtca4u
import versionnumbers as vn

mtca4u.set_dmap_location("deviceInformation/exampleCrate.dmap")

class TestMappedPCIEDevice(unittest.TestCase):
  # TODO: Refactor to take care of the harcoded values used for comparisions
  def testRead(self):
    self.__prepareDataOnCards()
    
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/mtcadummy.map")
    self.__testRead(device, "", device.read)
    
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/modular_mtcadummy.map")
    self.__testRead(device, "BOARD", device.read)
    
    device = mtca4u.Device("CARD_WITH_MODULES")
    self.__testRead(device, "BOARD", device.read)
    
    device = mtca4u.Device("CARD_WITH_OUT_MODULES")
    self.__testRead(device, "", device.read)

  def testWrite(self):
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/mtcadummy.map")
    self.__testWrite(device, "", device.write)
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/modular_mtcadummy.map")
    self.__testWrite(device, "BOARD", device.write)
    
    device = mtca4u.Device("CARD_WITH_MODULES")
    self.__testWrite(device, "BOARD", device.write)
    
    device = mtca4u.Device("CARD_WITH_OUT_MODULES")
    self.__testWrite(device, "", device.write)

  def testReadRaw(self):
    self.__prepareDataOnCards()
    
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/mtcadummy.map")
    self.__testRead(device, "", device.read_raw)
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/modular_mtcadummy.map")
    self.__testRead(device, "BOARD", device.read_raw)
    
    device = mtca4u.Device("CARD_WITH_OUT_MODULES")
    self.__testRead(device, "", device.read_raw)
    device = mtca4u.Device("CARD_WITH_MODULES")
    self.__testRead(device, "BOARD", device.read_raw)
    
  def testwriteRaw(self):
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/mtcadummy.map")
    self.__testWrite(device, "", device.write_raw)
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/modular_mtcadummy.map")
    self.__testWrite(device, "BOARD", device.write_raw)
    
    device = mtca4u.Device("CARD_WITH_OUT_MODULES")
    self.__testWrite(device, "", device.write_raw)
    device = mtca4u.Device("CARD_WITH_MODULES")
    self.__testWrite(device, "BOARD", device.write_raw)

  def testreadDMARaw(self):
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/mtcadummy.map")
    self.__testreadDMARaw(device, "")
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/modular_mtcadummy.map")
    self.__testreadDMARaw(device, "BOARD")
    
    device = mtca4u.Device("CARD_WITH_OUT_MODULES")
    self.__testreadDMARaw(device, "")
    device = mtca4u.Device("CARD_WITH_MODULES")
    self.__testreadDMARaw(device, "BOARD")

  def testReadSequences(self):      
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/mtcadummy.map")
    self.__testSequences(device, "")
    device = mtca4u.Device("/dev/mtcadummys1","deviceInformation/modular_mtcadummy.map")
    self.__testSequences(device, "BOARD")

    device = mtca4u.Device("CARD_WITH_OUT_MODULES")
    self.__testSequences(device, "")
    device = mtca4u.Device("CARD_WITH_MODULES")
    self.__testSequences(device, "BOARD")
# http://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python
  def testGetInfo(self):
      from StringIO import StringIO
      expectedString = "mtca4uPy v" +vn.moduleVersion + ", linked with mtca4u-deviceaccess v"+ vn.mappedDeviceVersion
      outStream = StringIO()
      mtca4u.get_info(outStream)
      returnedString = outStream.getvalue().strip()
      self.assertTrue(expectedString == returnedString)

  def testException(self):
    device = mtca4u.mtca4udeviceaccess.createDevice("/dev/mtcadummys1",
                                                    "deviceInformation/modular_mtcadummy.map")
    array = numpy.array([1, 2, 3, 4], dtype = numpy.int32)
    self.assertRaisesRegexp(RuntimeError, "size to write is more than the supplied array size", 
                        device.writeRaw, 8, array, (array.size * 4) + 1 , 0)

  def testDeviceCreation(self):
    self.assertRaisesRegexp(RuntimeError, "Unknown device alias", mtca4u.Device, "NON_EXISTENT_ALIAS_NAME") 

    self.assertRaisesRegexp(RuntimeError, "Cannot open file \"NON_EXISTENT_MAPFILE\"", mtca4u.Device,
                            "NON_EXISTENT_MAPFILE", "NON_EXISTENT_MAPFILE") 
    self.assertRaisesRegexp(SyntaxError, "Device called with incorrect number of parameters.", 
                            mtca4u.Device)
    self.assertRaisesRegexp(SyntaxError, "Device called with incorrect number of parameters.", 
                            mtca4u.Device, "BogusText", "BogusText", "BogusText")
    
  """
  The idea here is to preset data on registers that is then  read in and
  verified later. The following registers on each card are set: 
  - WORD_STATUS (Offset: 8)
  - WORD_CLK_MUX (Offset: 32)
  - WORD_INCOMPLETE_2 (Offset: 100)
  The memory map for each device has been kept identical. The map files all
  contain unique register names which are at the same address on each card
  (despite being in different modules on individual cards).
  A copy of the data that gets written is stored in these variables:
  - word_status_content
  - word_clk_mux_content
  - word_incomplete_content 
  """
  def __prepareDataOnCards(self):
    self.__prepareDataToWrite()
    self.__writeDataToDevices()
  

  def __prepareDataToWrite(self):
    self.word_status_content = self.__createRandomArray(1)
    self.word_clk_mux_content = self.__createRandomArray(4)
    self.word_incomplete_2_content = numpy.array([544], dtype = numpy.int32)
    
  def __writeDataToDevices(self):
        # Test Read from a module register
    # set up the register with a known values
    device = mtca4u.mtca4udeviceaccess.createDevice("/dev/mtcadummys1",
                                                    "deviceInformation/modular_mtcadummy.map")
    self.__preSetValuesOnCard(device)
    device = mtca4u.mtca4udeviceaccess.createDevice("/dev/llrfdummys4",
                                                    "deviceInformation/mtcadummy.map")
    self.__preSetValuesOnCard(device)
  
  def __createRandomArray(self, arrayLength):
    array = numpy.random.random_integers(0, 1073741824, arrayLength)
    return array.astype(numpy.int32)
  
  def __preSetValuesOnCard(self, device):
    WORD_STATUS_RegisterOffset = 8    
    bytesToWrite = self.word_status_content.size * 4 #  1 32 bit word -> 1 element
    bar = 0
    device.writeRaw(WORD_STATUS_RegisterOffset, self.word_status_content,
                    bytesToWrite, bar)
    
    WORD_CLK_MUX_REG_Offset = 32
    bytesToWrite = self.word_clk_mux_content.size * 4
    bar = 0 
    device.writeRaw(WORD_CLK_MUX_REG_Offset, self.word_clk_mux_content, bytesToWrite, bar)
    
    WORD_INCOMPLETE_2_REG_OFFSET = 100
    bytesToWrite = self.word_incomplete_2_content.size * 4
    bar = 0
    dataToSet = numpy.array([544], dtype = numpy.int32) # FP representation
                                                        # of 2.15
    device.writeRaw(WORD_INCOMPLETE_2_REG_OFFSET, self.word_incomplete_2_content, bytesToWrite, bar)

  def __testRead(self, device, module, readCommand):
    
    dtype = self.__getDtypeToUse(device, readCommand)
    
    word_status_content = self.word_status_content.astype(dtype)
    word_clk_mux_content = self.word_clk_mux_content.astype(dtype)
    
    # Test the read from module functionality 
    
    readInValues = readCommand(str(module), "WORD_STATUS")
    self.assertTrue(readInValues.dtype == dtype)
    self.assertTrue(numpy.array_equiv(readInValues, word_status_content))
    
    # This section checks the read register code for the Device class
    
    # check if function reads values correctly
    # Run this only for device.read
    if readCommand == device.read:
      readInValues = readCommand(str(module), "WORD_INCOMPLETE_2")
      self.assertTrue(readInValues.dtype == dtype)
      self.assertTrue(readInValues.tolist() == [2.125])
    
    readInValues = readCommand(str(module), "WORD_CLK_MUX")
    self.assertTrue(readInValues.dtype == dtype)
    self.assertTrue(numpy.array_equiv(readInValues, word_clk_mux_content))
    
    readInValues = readCommand(str(module), "WORD_CLK_MUX", 1)
    self.assertTrue(readInValues[0] == word_clk_mux_content[0])
    
    readInValues = readCommand(str(module), "WORD_CLK_MUX", 1, 2)
    self.assertTrue(readInValues[0] == word_clk_mux_content[2])
    
    readInValues = readCommand(str(module), "WORD_CLK_MUX", 0, 2)
    self.assertTrue(numpy.array_equiv(readInValues, word_clk_mux_content[2:]))
    
    # check for corner cases
    # Register Not Found
    # hack
    exceptionMessage = self.__returnRegisterNotFoundExceptionMsg(module, "BAD_REGISTER_NAME")
      
    self.assertRaisesRegexp(RuntimeError, exceptionMessage, readCommand, str(module), 
                            "BAD_REGISTER_NAME")


    # Num of elements specified  is more than register size
    registerName = "WORD_CLK_MUX"
    elementsToRead = 5
    offset = 2
    readInValues = readCommand(str(module) ,registerName, elementsToRead, offset)
    self.assertTrue(readInValues.dtype == dtype)
    self.assertTrue(numpy.array_equiv(readInValues, word_clk_mux_content[2:]))
    
    # bad value for number of elements
    self.assertRaisesRegexp(ValueError, "negative dimensions are not allowed",
                             readCommand, str(module), registerName, 
                             numberOfElementsToRead=-1)
    
    # offset exceeds register size
    offset = 5
    elementsToRead = 5
    self.assertRaisesRegexp(ValueError, "Element index: 5 incorrect. Valid index"
                            " range is \[0-3\]", readCommand,  str(module),
                            registerName, elementIndexInRegister = offset) 
    self.assertRaisesRegexp(ValueError, "Element index: 5 incorrect. Valid index"
                            " range is \[0-3\]", readCommand,  str(module),
                            registerName, elementsToRead, offset)
    self.assertRaisesRegexp(OverflowError, "can't convert negative value to"
                            " unsigned", readCommand,  str(module),
                            registerName, elementIndexInRegister = -1)
    
  def __testWrite(self, device, module, writeCommand ):
    
    module = str(module)
    dtype = self.__getDtypeToUse(device, writeCommand)
    
    if writeCommand == device.write:
      readCommand = device.read
    else:
      readCommand = device.read_raw
      
    word_status_content = self.__createRandomArray(1).astype(dtype)
    word_clk_mux_content = self.__createRandomArray(4).astype(dtype)
     
    writeCommand(module, "WORD_STATUS", word_status_content)
    readInValues = readCommand(module, "WORD_STATUS")
    self.assertTrue(readInValues.dtype == dtype)
    self.assertTrue(numpy.array_equiv(readInValues, word_status_content))
     
     
     # These set of commands will be run for Device.write only
     
    word_incomplete_register = "WORD_INCOMPLETE_2"
    if writeCommand == device.write:
     # write to WORD_INCOMPLETE_2, this is 13 bits wide and supports 8
     # fractional bits
     # check the write functionality
     # check functionalty when using dtype numpy.float32
      writeCommand(module, word_incomplete_register, 
                 numpy.array([2.125], dtype))
      readInValue = readCommand(module, word_incomplete_register)
      self.assertTrue(readInValue.dtype == numpy.float32)
      self.assertTrue(readInValue.tolist() == [2.125])
     
     # check functionalty when using dtype numpy.float64
      writeCommand(module, word_incomplete_register, 
                 numpy.array([3.125], dtype = numpy.float64))
      readInValue = readCommand(module, word_incomplete_register)
      self.assertTrue(readInValue.dtype == dtype)
      self.assertTrue(readInValue.tolist() == [3.125])
    
      # check functionalty when using dtype numpy.int32
      writeCommand(module, word_incomplete_register, 
                 numpy.array([2], dtype = numpy.int32))
      readInValue = readCommand(module, word_incomplete_register)
      self.assertTrue(readInValue.dtype == dtype)
      self.assertTrue(readInValue.tolist() == [2.])
    
      # check functionalty when using dtype numpy.int64
      writeCommand(module, word_incomplete_register, 
                 numpy.array([25], dtype = numpy.int64))
      readInValue = readCommand(module, word_incomplete_register)
      self.assertTrue(readInValue.dtype == dtype)
      self.assertTrue(readInValue.tolist() == [15.99609375])  # This is the 
                                                            # valid fp converted
                                                            # value of int 25 
                                                            # for this reg
 
      writeCommand(module, word_incomplete_register,[2.5])
      readInValues = readCommand(module, word_incomplete_register)
      self.assertTrue(readInValues.tolist() == [2.5])
      
      # continue tests for checking if method accepts int/float/list/numpyarray as valid dataToWrite
      # input a list
      
      writeCommand(module, "WORD_CLK_MUX", word_status_content, 1)
      readInValues = readCommand(module, "WORD_CLK_MUX", 1, 1)
      self.assertTrue(numpy.array_equiv(readInValues, word_status_content))
      
      writeCommand(module, word_incomplete_register, 3.5)
      readInValues = readCommand(module, word_incomplete_register)
      self.assertTrue(readInValues.tolist() == [3.5])
       
      writeCommand(module, word_incomplete_register, 14)
      readInValues = readCommand(module, word_incomplete_register)
      self.assertTrue(readInValues.tolist() == [14])
      
      writeCommand(module, "WORD_CLK_MUX", 5)
      readInValues = readCommand(module, "WORD_CLK_MUX", 1, 0)
      self.assertTrue(readInValues.tolist() == [5])
      
      self.assertRaisesRegexp(RuntimeError, "Data format used is unsupported",
                              writeCommand, module,  word_incomplete_register, 
                              "")
 
    
    
     # Test for Unsupported dtype eg. dtype = numpy.int8 
      self.assertRaisesRegexp(RuntimeError, "Data format used is unsupported",
                              writeCommand, module,  word_incomplete_register, 
                              numpy.array([2], dtype = numpy.int8))
    
    # check offset functionality
    writeCommand(module, "WORD_CLK_MUX", word_clk_mux_content)
    readInValues = readCommand(module, "WORD_CLK_MUX")
    self.assertTrue(numpy.array_equiv(readInValues, word_clk_mux_content))
    
    word_clk_mux_register = "WORD_CLK_MUX"
    writeCommand(module, word_clk_mux_register, word_clk_mux_content[0:2], 
                 elementIndexInRegister = 2)
    readInValue = readCommand(module, word_clk_mux_register,
                              elementIndexInRegister = 2)
    self.assertTrue(readInValue.dtype == dtype)
    self.assertTrue(numpy.array_equiv(readInValue, word_clk_mux_content[0:2]))
    # Check corner cases
      
    # Bogus register name
    exceptionMessage = self.__returnRegisterNotFoundExceptionMsg(module, "BAD_REGISTER_NAME")
    self.assertRaisesRegexp(RuntimeError, exceptionMessage, writeCommand, module,  
                            "BAD_REGISTER_NAME", 
                            numpy.array([2.125], dtype = dtype))
    
    # supplied array size exceeds register capacity
    self.assertRaisesRegexp(RuntimeError, "Data size exceed register size",
                             writeCommand, module, word_incomplete_register, 
                             word_clk_mux_content)
     
    # supplied offset exceeds register span
    self.assertRaisesRegexp(ValueError, "Element index: 1 incorrect."
                            " Valid index is 0", writeCommand, module,  
                            word_incomplete_register, word_clk_mux_content,
                            elementIndexInRegister=1)
    # write nothing
    initialValue = readCommand(module, "WORD_STATUS")
    writeCommand(module,"WORD_STATUS", numpy.array([], dtype = dtype))
    valueAfterEmptyWrite = readCommand(module, "WORD_STATUS")
    self.assertTrue(numpy.array_equiv(initialValue, valueAfterEmptyWrite))
    
  def __returnRegisterNotFoundExceptionMsg(self, module, registerName):
    if not str(module):
      exceptionMessage = "Cannot find register " + str(registerName) + \
                         " in map file: deviceInformation/mtcadummy.map"
    else:
      exceptionMessage = "Cannot find register " + str(module) + "." + str(registerName) + \
                    " in map file: deviceInformation/modular_mtcadummy.map"
  def __getDtypeToUse(self, device, command):
    if command == device.read or command == device.write:
      return numpy.float32
    elif command == device.read_raw or command == device.write_raw:
      return numpy.int32
    
  def __testreadDMARaw(self, device, module):
    module = str(module)
    # Set the parabolic values in the DMA region by writing 1 to WORD_ADC_ENA
    # register
    device.write(module, "WORD_ADC_ENA", numpy.array([1], dtype = numpy.float32))
                  
    # Read in the parabolic values from the function
    readInValues = device.read_dma_raw(module, "AREA_DMA_VIA_DMA", 
                                     numberOfElementsToRead= 10)
    self.assertTrue(readInValues.dtype == numpy.int32)
    self.assertTrue(readInValues.tolist() == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
  
    # Check offset read
    readInValues = device.read_dma_raw(module, "AREA_DMA_VIA_DMA", 
                                     numberOfElementsToRead=10,
                                     elementIndexInRegister=3)
    self.assertTrue(readInValues.dtype == numpy.int32)
    self.assertTrue(readInValues.tolist() == [9, 16, 25, 36, 49, 64, 81, 100, \
                                             121, 144])
  
    # corner cases:
    # bad register name
    exceptionText = self.__returnRegisterNotFoundExceptionMsg(module, "BAD_REGISTER_NAME")
    # bad element size
    # bad offset
    # FIXME: Not checking this; size of  AREA_DMA_VIA_DMA is big 1024 elements

  def __testSequences(self, device, module):
    module = str(module)
    # Basic Interface: Currently supports read of all sequences only
    #device.write("", "WORD_ADC_ENA", 1)
    # Arrange the data on the card:
    predefinedSequence = numpy.array([0x00010000,
                                      0x00030002,
                                      0x00050004,
                                      0x00070006,
                                      0x00090008,
                                      0x000b000a,
                                      0x000d000c,
                                      0x000f000e,
                                      0x00110010,
                                      0x00130012,
                                      0x00150014,
                                      0x00170016,
                                      0x00ff0018], dtype=numpy.int32)
    device.write_raw(module, 'AREA_DMAABLE', predefinedSequence)
    
    expectedMatrix = numpy.array([[0,  1,  2,  3],
                                  [4,  5,  6,  7],
                                  [8, 9, 10, 11],
                                  [12, 13, 14, 15],
                                  [16, 17, 18, 19],
                                  [20, 21, 22, 23]], dtype=numpy.float32)
    readInMatrix = device.read_sequences(module, 'DMA')
    self.assertTrue(numpy.array_equiv(readInMatrix, expectedMatrix))
    self.assertTrue(readInMatrix.dtype == numpy.float32)
  
if __name__ == '__main__':
    unittest.main()     

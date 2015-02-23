#! /usr/bin/python
import os
import sys
import  unittest
import numpy

# This is a hack for nw
sys.path.insert(0,os.path.abspath(os.curdir))
import mtcamappeddevice

class TestPCIEDevice(unittest.TestCase):

    def testCreatePCIEDevice(self):
        self.assertRaisesRegexp(RuntimeError, "Cannot open device: : No such "
                "file or directory", mtcamappeddevice.createDevice, "")
        self.assertRaisesRegexp(RuntimeError, "Cannot open device: "
                "some_non_existent_device: No such file or directory", mtcamappeddevice.createDevice,
                "some_non_existent_device")

    def testreadRaw(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4")
        wordCompilationRegOffset = 4
        preAllocatedArray = numpy.zeros(2, dtype = numpy.int32)
        bytesToRead = 12
        registerBar = 0

        self.assertRaisesRegexp(RuntimeError, "size to write is more than the "
        "supplied array size", device.readRaw, wordCompilationRegOffset,
        preAllocatedArray, bytesToRead, registerBar)


        badRegOffset = 563
        bytesToRead = 8

        self.assertRaisesRegexp(RuntimeError, "Cannot read data from device: "
        "/dev/llrfdummys4: Bad address", device.readRaw, badRegOffset,
        preAllocatedArray, bytesToRead, registerBar)


        # read in th default value from WORD_FIRMWARE register
        wordCompilationRegOffset = 4
        bytesToRead = 4

        device.readRaw(wordCompilationRegOffset, preAllocatedArray, bytesToRead,
                registerBar)
        self.assertTrue( 9 == preAllocatedArray[0])


        wordStatusRegOffset = 8
        dataArray = numpy.array([5, 9], dtype=numpy.int32)
        readInArray = numpy.zeros(2, dtype = numpy.int32)
        bytesToWrite = 8
        bytesToRead = 8

        device.writeRaw(wordStatusRegOffset, dataArray,
                bytesToWrite, registerBar)
        device.readRaw(wordStatusRegOffset, readInArray, bytesToRead,
                registerBar)

        self.assertTrue(readInArray.tolist() == dataArray.tolist())

    def testwriteRaw(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4")
        wordStatusRegOffset = 8
        infoToWrite = numpy.array([566,58], dtype = numpy.int32)
        bytesToWrite = 12
        registerBar = 0


        self.assertRaisesRegexp(RuntimeError, "size to write is more than the "
        "supplied array size", device.writeRaw, wordStatusRegOffset,
        infoToWrite, bytesToWrite, registerBar)
        
        badRegOffset = 5654
        bytesToWrite = 8
        
        self.assertRaisesRegexp(RuntimeError, "Cannot read data from device: "
        "/dev/llrfdummys4: Bad address", device.readRaw, badRegOffset,
        infoToWrite, bytesToWrite, registerBar)

        # test of write done in the above testcase TODO: make this proper


    def testWriteRawUsingRegName(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4") 
        registerName = "WORD_CLK_MUX"
        spaceToReadIn = numpy.zeros(4, dtype = numpy.int32)
        bytesToReadIn = 0 # 0 => read in the whole register 
        offset = 0 # start reading from the begining of the register
        self.assertRaisesRegexp(RuntimeError, "This method is not available for"
                " this device", device.writeRaw, registerName, spaceToReadIn,
                bytesToReadIn, offset)

    def testReadRawUsingRegName(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4")

        # setup WORD_CLK_MUX register with custom values
        regOffset = 32
        customData = numpy.array([1, 2, 3, 4], dtype = numpy.int32)
        bytesToWrite = 4*4 # 4 words
        bar = 0
        device.writeRaw(regOffset, customData, bytesToWrite, bar)


        registerName = "WORD_CLK_MUX"
        # array big enough to hold the whole register
        spaceToReadIn = numpy.zeros(4, dtype = numpy.int32)
        bytesToReadIn = 0 # 0 => read in the whole register
        offset = 0 # start reading from the begining of the register

        # read in the register (4 words long) using its name
        self.assertRaisesRegexp(RuntimeError, "This method is not available for"
        " this device", device.readRaw, registerName, spaceToReadIn,
        bytesToReadIn, offset)

    def testReadDMA(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4")
        # Set the WORD_ADC_ENA reg to 1; This sets the first 25 words of the
        # DMA memory area to a prabolically increasing set of values; The offset
        # for the WORD_ADC_ENA register is 68

        wordAdcEnaRegOffset = 68
        bytesToWrite = 4 # i.e one word
        registerBar = 0
        dataArray = numpy.array([1], dtype = numpy.int32)
        device.writeRaw(wordAdcEnaRegOffset, dataArray, bytesToWrite,
               registerBar) # the DMA area would be set after this

         # TODO: Use a ;loop later
        expectedDataArray = numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100,
            121, 144, 169, 196, 225, 256,  289, 324, 361, 400, 441, 484, 529,
            576], dtype = numpy.int32)


        # read the DMA area which has been set with values
        dmaArea = 0
        readInArray = numpy.zeros(25, dtype = numpy.int32)
        bytesToRead = 25 * 4 # 25 words

        device.readDMA(dmaArea, readInArray, bytesToRead)

        self.assertTrue(readInArray.tolist() == expectedDataArray.tolist())


    def testReadDMAUsingRegName(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4")         
        dmaRegName = "AREA_DMAABLE"
        dataToRead = numpy.zeros(10, dtype = numpy.int32)
        bytesToRead = 10 * 4
        offset = 0
        self.assertRaisesRegexp(RuntimeError, "This method is not available for"
                " this device", device.readDMA, dmaRegName, dataToRead,
                bytesToRead, offset)



    def testWriteDMA(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4") 
        #TODO: Use loop later
        #dataToWrite = numpy.array([576, 529, 484, 441, 400, 361, 324, 289, 256,
            #225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0],
            #dtype = numpy.int32)
        #dmaAreaAddress = 0
        #bytesToWrite = 25 * 4 # 25 entries inside dataToWrite
        #device.writeDMA(dmaAreaAddress, dataToWrite, bytesToWrite)

        #dataToRead = numpy.zeros(25, dtype = numpy.int32) # Space for content to read from
                                                          # DMA Area
        #bytesToRead = 25*4
        #device.readDMA(dmaAreaAddress, dataToRead, bytesToRead)

        #self.assertTrue(dataToRead.tolist() == dataToWrite.tolist())

        dmaAreaAddress = 0
        dataToWrite = numpy.array([1,2], dtype = numpy.int32)
        bytesToWrite = 2*4
        self.assertRaisesRegexp(RuntimeError, "Operation not supported yet",
                device.writeDMA, dmaAreaAddress, dataToWrite, bytesToWrite)
       

    def testWriteDMAThroughRegisterName(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4")           
        registerName = "AREA_DMAABLE"
        dataArray = numpy.zeros(1, dtype = numpy.int32)
        bytesToRead = 1 * 4 # one word
        offset = 0
        self.assertRaisesRegexp(RuntimeError, "This method is not available for"
                " this device", device.writeDMA, registerName, dataArray,
                bytesToRead, offset)


if __name__ == '__main__':
    unittest.main()

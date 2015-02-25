#! /usr/bin/python

import os
import sys
import  unittest
import numpy

# This is a hack for nw
sys.path.insert(0,os.path.abspath(os.curdir))
import mtcamappeddevice

class TestDummyDevice(unittest.TestCase):

    def testCreateDummyDevice(self):
        self.assertRaisesRegexp(RuntimeError, "Mapped Dummy Device expects"
                " first and second parameters to be the same map file"
                , mtcamappeddevice.createDevice,
                "mapfiles/mtcadummy.map", "someBogusMapFile.map")

    def testreadRaw(self):
        #TODO: Move the mapfile location to a global common variable
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map",
                "mapfiles/mtcadummy.map")
        wordCompilationRegOffset = 4
        preAllocatedArray = numpy.zeros(2, dtype = numpy.int32)
        bytesToRead = 12
        registerBar = 0

        self.assertRaisesRegexp(RuntimeError, "size to write is more than the "
        "supplied array size", device.readRaw, wordCompilationRegOffset,
        preAllocatedArray, bytesToRead, registerBar)


        badRegOffset = 563
        bytesToRead = 8

        self.assertRaisesRegexp(RuntimeError, "Invalid address offset 563 in"
                " bar 0.Caught out_of_range exception: vector::_M_range_check"
        , device.readRaw, badRegOffset,
        preAllocatedArray, bytesToRead, registerBar)


        # read in th default value from WORD_FIRMWARE register
        wordCompilationRegOffset = 4
        bytesToRead = 4

        device.readRaw(wordCompilationRegOffset, preAllocatedArray, bytesToRead,
                registerBar)
        self.assertTrue( 0 == preAllocatedArray[0])


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
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map",
                "mapfiles/mtcadummy.map")
        wordStatusRegOffset = 8
        infoToWrite = numpy.array([566,58], dtype = numpy.int32)
        bytesToWrite = 12
        registerBar = 0


        self.assertRaisesRegexp(RuntimeError, "size to write is more than the "
        "supplied array size", device.writeRaw, wordStatusRegOffset,
        infoToWrite, bytesToWrite, registerBar)
        
        badRegOffset = 5654
        bytesToWrite = 8
        
        self.assertRaisesRegexp(RuntimeError, "Invalid address offset 5654 in"
                " bar 0.Caught out_of_range exception: vector::_M_range_check"
        , device.readRaw, badRegOffset,
        infoToWrite, bytesToWrite, registerBar)

    def testWriteRawUsingRegName(self):
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map",
                "mapfiles/mtcadummy.map")
        registerName = "WORD_CLK_MUX"
       
        # pre set values in the register
        registerOffset = 32
        dataToSetInRegister = numpy.array([15, 14, 13, 12], dtype = numpy.int32)
        bytesToSet = 4 * 4 # 4 words
        bar = 0

        spaceToReadIn = numpy.zeros(4, dtype = numpy.int32)
        bytesToReadIn = 0 # 0 => read in the whole register 
        offset = 0 # start reading from the begining of the register
        device.writeRaw(registerName, dataToSetInRegister, bytesToReadIn, offset)
        device.readRaw(registerOffset, spaceToReadIn, bytesToSet, bar)

        self.assertTrue(spaceToReadIn.tolist() == dataToSetInRegister.tolist())

    def testReadRawUsingRegName(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4",
        "mapfiles/mtcadummy.map")
        registerName = "WORD_CLK_MUX"
       
        # pre set values in the register
        registerOffset = 32
        dataToSetInRegister = numpy.array([5, 4, 3, 2], dtype = numpy.int32)
        bytesToSet = 4 * 4 # 4 words
        bar = 0
        device.writeRaw(registerOffset, dataToSetInRegister, bytesToSet, bar)

        spaceToReadIn = numpy.zeros(4, dtype = numpy.int32)
        bytesToReadIn = 0 # 0 => read in the whole register 
        offset = 0 # start reading from the begining of the register
        device.readRaw(registerName, spaceToReadIn, bytesToReadIn, offset)

        self.assertTrue(spaceToReadIn.tolist() == dataToSetInRegister.tolist())

    def testreadDMA(self):
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map",
                "mapfiles/mtcadummy.map")
        # Read DMA internally a wrapper around readArea in the API

        wordStatusRegOffset = 8
        dataArray = numpy.array([5, 9], dtype=numpy.int32)
        readInArray = numpy.zeros(2, dtype = numpy.int32)
        bytesToWrite = 8
        bytesToRead = 8
        registerBar = 0

        device.writeRaw(wordStatusRegOffset, dataArray,
                bytesToWrite, registerBar)
        device.readDMA(wordStatusRegOffset, readInArray, bytesToRead)

        self.assertTrue(readInArray.tolist() == dataArray.tolist())

    def testReadDMAUsingRegName(self):
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map",
                "mapfiles/mtcadummy.map")
        #TODO: Use loop later
        dataToWrite = numpy.array([576, 529, 484, 441, 400, 361, 324, 289, 256,
            225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0],
            dtype = numpy.int32)
       
        areaDMAABLEOffset = 0
        bytesToWrite = 25 * 4 # 25 entries inside dataToWrite
        bar = 2
        device.writeRaw(areaDMAABLEOffset, dataToWrite, bytesToWrite, bar)

        dataToRead = numpy.zeros(25, dtype = numpy.int32) # Space for content to read from
                                                          # DMA Area
        bytesToRead = 0 # read all
        offset = 0
        dmaRegName = "AREA_DMAABLE"
        device.readDMA(dmaRegName, dataToRead, bytesToRead, offset)

        self.assertTrue(dataToRead.tolist() == dataToWrite.tolist())
       
    def testWriteDMA(self):
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map",
                "mapfiles/mtcadummy.map")
        dmaAreaAddress = 0
        dataToWrite = numpy.array([1,2], dtype = numpy.int32)
        bytesToWrite = 2*4
        self.assertRaisesRegexp(RuntimeError, "DummyDevice::writeDMA is not"
                " implemented yet.", device.writeDMA, dmaAreaAddress, 
                dataToWrite, bytesToWrite)


    def testWriteDMAThroughRegisterName(self):
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map",
                "mapfiles/mtcadummy.map")
        registerName = "AREA_DMA_VIA_DMA"
        dataArray = numpy.zeros(1, dtype = numpy.int32)
        bytesToRead = 1 * 4 # one word
        offset = 0
        self.assertRaisesRegexp(RuntimeError, "DummyDevice::writeDMA is not"
                " implemented yet.", device.writeDMA, registerName, dataArray,
                bytesToRead, offset)

if __name__ == '__main__':
    unittest.main()

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

        # Set data into the multiword register: WORD_CLK_MUX
        registerName = "WORD_CLK_MUX"
        dataToSetInRegister = numpy.array([15, 14, 13, 12], dtype = numpy.int32)
        setAllWordsInRegister = 0
        offset = 0
        device.writeRaw(registerName, dataToSetInRegister,
                setAllWordsInRegister, offset)
       
        # Verify if the values have been written
        WORD_CLK_MUX_OFFSET = 32
        spaceToReadIn = numpy.zeros(4, dtype = numpy.int32)
        bytesToReadIn = 4  * 4 # 0 => read in the whole register 
        bar = 0 # start reading from the begining of the register
        device.readRaw(WORD_CLK_MUX_OFFSET, spaceToReadIn, bytesToReadIn, bar)

        self.assertTrue(spaceToReadIn.tolist() == dataToSetInRegister.tolist())

    def testReadRawUsingRegName(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4",
        "mapfiles/mtcadummy.map")
        registerName = "WORD_CLK_MUX"

        # pre set values in the register
        dataToSetInRegister = numpy.array([5, 4, 3, 2], dtype = numpy.int32)
        bytesToSet = 4 * 4 # 4 words
        bar = 0
        device.writeRaw(registerName, dataToSetInRegister, bytesToSet, bar)

        # read in the contents using the Registername
        spaceToReadIn = numpy.zeros(4, dtype = numpy.int32)
        readAllWordsInRegister = 0 # 0 => read in the whole register 
        offset = 0 # start reading from the begining of the register
        device.readRaw(registerName, spaceToReadIn, readAllWordsInRegister, offset)

        self.assertTrue(spaceToReadIn.tolist() == dataToSetInRegister.tolist())

    def testreadDMA(self):
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map",
                "mapfiles/mtcadummy.map")
        # Read DMA internally a wrapper around readArea in the API
        # DummyDevice readArea and readDMA does the same things
        # Currently dummy device does not have a DMA region

        # For the test we write to the WORD_CLK_MUX register and read contents
        # back in using the readDMA function. WORD_CLK_MUX is a multi word
        # register (accomodating 4 words)

        WORD_CLK_MUX_OFFSET = 32
        dataArray = numpy.array([15, 19, 45, 20], dtype=numpy.int32)
        bytesToWrite = 4 * 4 # 4 words
        registerBar = 0

        # Fill content of the dataArray into WORD_CLK_MUX 
        device.writeRaw(WORD_CLK_MUX_OFFSET, dataArray,
                bytesToWrite, registerBar)


        # Verify if readDMA can read back the written values
        spaceForValuesTobeReadIn = numpy.zeros(4, dtype = numpy.int32)
        bytesToRead = bytesToWrite
        device.readDMA(WORD_CLK_MUX_OFFSET, spaceForValuesTobeReadIn, bytesToRead)
        self.assertTrue(spaceForValuesTobeReadIn.tolist() == dataArray.tolist())

    def testReadDMAUsingRegName(self):
        # The test content should be similar to readDMA(/readRaw - because internally
        # both methods does the same thing)
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map",
                "mapfiles/mtcadummy.map")

        # Since readDMA is the
        # same as readRaw from a register, we test this on a normal multiword
        # register WORD_CLK_MUX
        registerName = "AREA_DMA_VIA_DMA"
        AREA_DMA_VIA_DMA_OFFSET = 0
        dataToSetInRegister = numpy.array([85, 94, 23, 12], dtype = numpy.int32)
        bytesToWrite = 4 * 4 # 4 words
        offset = 13

        # use writeRaw to put in the test values to AREA_DMA_VIA_DMA.
        device.writeRaw(AREA_DMA_VIA_DMA_OFFSET, dataToSetInRegister,
                bytesToWrite, offset)

        # readIn what was written and verify contents:
        spaceToReadInRegister = numpy.zeros(4, dtype = numpy.int32)
        bytesToRead = 4 * 4 # we would want to read only the first 4 words, not
                            # the whole register (AREA_DMA_VIA_DMA is 4096 bytes long)
        offset = 0
        device.readDMA(registerName, spaceToReadInRegister, bytesToRead,
                offset)

        self.assertTrue(dataToSetInRegister.tolist() == spaceToReadInRegister.tolist())
       
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

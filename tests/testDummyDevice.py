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
        self.assertRaisesRegexp(RuntimeError, "Cannot open device: : No such "
                "file or directory", mtcamappeddevice.createDevice, "")
        self.assertRaisesRegexp(RuntimeError, "Cannot open device: "
                "some_non_existent_device: No such file or directory", mtcamappeddevice.createDevice,
                "some_non_existent_device")
        self.assertRaisesRegexp(RuntimeError, "Cannot open device: "
                "../mapfiles/mtcadummy.map1: No such file or directory", mtcamappeddevice.createDevice,
                "../mapfiles/mtcadummy.map1")

    def testreadRaw(self):
        #TODO: Move the mapfile location to a global common variable
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map")
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
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map")
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

    def testreadDMA(self):
        # Read DMA internally a wrapper around readArea in the API
        # Repeat the same tests as for readRaw
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map")

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

    def testWriteDMA(self):
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map")
        dmaAreaAddress = 0
        dataToWrite = numpy.array([1,2], dtype = numpy.int32)
        bytesToWrite = 2*4
        self.assertRaisesRegexp(RuntimeError, "DummyDevice::writeDMA is not"
                " implemented yet.", device.writeDMA, dmaAreaAddress,
                dataToWrite, bytesToWrite)


    def testWriteDMAThroughRegisterName(self):
        device = mtcamappeddevice.createDevice("mapfiles/mtcadummy.map")
        registerName = "AREA_DMAABLE"
        dataArray = numpy.zeros(1, dtype = numpy.int32)
        bytesToRead = 1 * 4 # one word
        offset = 0
        self.assertRaisesRegexp(RuntimeError, "This method is not available for"
                " this device", device.writeDMA, registerName, dataArray,
                bytesToRead, offset)
if __name__ == '__main__':
    unittest.main()

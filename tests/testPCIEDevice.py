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

    def testReadArea(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4")
        wordCompilationRegOffset = 4
        preAllocatedArray = numpy.zeros(2, dtype = numpy.int32)
        bytesToRead = 12
        registerBar = 0

        self.assertRaisesRegexp(RuntimeError, "size to write is more than the "
        "supplied array size", device.readArea, wordCompilationRegOffset,
        preAllocatedArray, bytesToRead, registerBar)


        badRegOffset = 563
        bytesToRead = 8

        self.assertRaisesRegexp(RuntimeError, "Cannot read data from device: "
        "/dev/llrfdummys4: Bad address", device.readArea, badRegOffset,
        preAllocatedArray, bytesToRead, registerBar)


        # read in th default value from WORD_FIRMWARE register
        wordCompilationRegOffset = 4
        bytesToRead = 4

        device.readArea(wordCompilationRegOffset, preAllocatedArray, bytesToRead,
                registerBar)
        self.assertTrue( 9 == preAllocatedArray[0])


        wordStatusRegOffset = 8
        dataArray = numpy.array([5, 9], dtype=numpy.int32)
        readInArray = numpy.zeros(2, dtype = numpy.int32)
        bytesToWrite = 8
        bytesToRead = 8

        device.writeArea(wordStatusRegOffset, dataArray,
                bytesToWrite, registerBar)
        device.readArea(wordStatusRegOffset, readInArray, bytesToRead,
                registerBar)

        print readInArray
        print dataArray
        self.assertTrue(readInArray.tolist() == dataArray.tolist())

    def testWriteArea(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4")
        wordStatusRegOffset = 8
        infoToWrite = numpy.array([566,58], dtype = numpy.int32)
        bytesToWrite = 12
        registerBar = 0


        self.assertRaisesRegexp(RuntimeError, "size to write is more than the "
        "supplied array size", device.writeArea, wordStatusRegOffset,
        infoToWrite, bytesToWrite, registerBar)
        
        badRegOffset = 5654
        bytesToWrite = 8
        
        self.assertRaisesRegexp(RuntimeError, "Cannot read data from device: "
        "/dev/llrfdummys4: Bad address", device.readArea, badRegOffset,
        infoToWrite, bytesToWrite, registerBar)

        # test of write done in the above testcase TODO: make this proper

    def test(self):
        self.assertTrue(True)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()

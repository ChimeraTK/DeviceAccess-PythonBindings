#! /usr/bin/python
import os
import sys
import  unittest
import numpy

# This is a hack for nw
sys.path.insert(0,os.path.abspath(os.curdir))
import mtcamappeddevice

class TestMappedPCIEDevice(unittest.TestCase):

    def testCreateMappedPCIEDevice(self):
        self.assertRaisesRegexp(RuntimeError, "Cannot open device: : No such "
                "file or directory", mtcamappeddevice.createDevice, "", "")



    def testreadRaw(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4",
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
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4",
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

        self.assertRaisesRegexp(RuntimeError, "Cannot read data from device: "
        "/dev/llrfdummys4: Bad address", device.readRaw, badRegOffset,
        infoToWrite, bytesToWrite, registerBar)

        # test of write done in the above testcase TODO: make this proper


    def testWriteRawUsingRegName(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4",
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

    def testReadDMA(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4",
        "mapfiles/mtcadummy.map")
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
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4",
        "mapfiles/mtcadummy.map")

        wordAdcEnaRegOffset = 68
        bytesToWrite = 4 # i.e one word
        registerBar = 0
        dataArray = numpy.array([1], dtype = numpy.int32)
        device.writeRaw(wordAdcEnaRegOffset, dataArray, bytesToWrite,
               registerBar) # the DMA area would be set after this

        dmaRegName = "AREA_DMA_VIA_DMA"
        dataToRead = numpy.zeros(25, dtype = numpy.int32)
        bytesToRead = 25 * 4
        offset = 0
        device.readDMA(dmaRegName, dataToRead, bytesToRead, offset)

        expectedDataArray = numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100,
            121, 144, 169, 196, 225, 256,  289, 324, 361, 400, 441, 484, 529,
            576], dtype = numpy.int32)

        self.assertTrue(dataToRead.tolist() == expectedDataArray.tolist())

    def testWriteDMA(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4",
        "mapfiles/mtcadummy.map")

        dmaAreaAddress = 0
        dataToWrite = numpy.array([1,2], dtype = numpy.int32)
        bytesToWrite = 2*4
        self.assertRaisesRegexp(RuntimeError, "Operation not supported yet",
                device.writeDMA, dmaAreaAddress, dataToWrite, bytesToWrite)


    def testWriteDMAThroughRegisterName(self):
        device = mtcamappeddevice.createDevice("/dev/llrfdummys4",
        "mapfiles/mtcadummy.map")
        registerName = "AREA_DMA_VIA_DMA"
        dataArray = numpy.zeros(1, dtype = numpy.int32)
        bytesToRead = 1 * 4 # one word
        offset = 0
        self.assertRaisesRegexp(RuntimeError, "Operation not supported yet"
                , device.writeDMA, registerName, dataArray,
                bytesToRead, offset)

if __name__ == '__main__':
    unittest.main()

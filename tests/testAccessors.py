#!/usr/bin/env python3
import sys
import unittest
import numpy as np
import os


# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module.
sys.path.insert(0, os.path.abspath(os.curdir))
import deviceaccess as da


class TestAccessors(unittest.TestCase):

    def testScalar(self):

        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        dev = da.Device("CARD_WITH_MODULES")
        acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        dev.open()
        reference = [99]

        acc.set(reference)
        acc.write()

        otherAcc = dev.getScalarRegisterAccessor(
            np.int32, "ADC/WORD_CLK_CNT_1")
        otherAcc.read()
        self.assertTrue(reference == otherAcc.view())
        dev.close()

    def testOneD(self):

        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        dev = da.Device("CARD_WITH_MODULES")
        acc = dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        dev.open()
        elements = acc.getNElements()
        reference = [i+42 for i in range(elements)]

        acc.set(reference)
        acc.write()

        otherAcc = dev.getOneDRegisterAccessor(
            np.int32, "BOARD/WORD_CLK_MUX")
        otherAcc.read()
        self.assertTrue(np.array_equal(reference, otherAcc.view()))
        dev.close()

    def testTwoD(self):

        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        dev = da.Device("CARD_WITH_MODULES")
        acc = dev.getTwoDRegisterAccessor(np.int32, "BOARD/DMA")
        dev.open()
        channels = acc.getNChannels()
        elementsPerChannel = acc.getNElementsPerChannel()
        reference = [
            [i*j+i+j+12 for j in range(elementsPerChannel)] for i in range(channels)]

        acc.set(reference)
        acc.write()

        otherAcc = dev.getTwoDRegisterAccessor(np.int32, "BOARD/DMA")
        otherAcc.read()
        self.assertTrue(np.array_equal(reference, otherAcc.view()))
        dev.close()

    def testTypes(self):
        # TODO test all the different userTypes from the Cpp lib
        pass


if __name__ == '__main__':
    unittest.main()

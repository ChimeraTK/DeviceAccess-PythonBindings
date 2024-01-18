#!/usr/bin/env python3
import sys
import unittest
import numpy as np
import os

# fmt: off
# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module. Formatting is switched off,
# so the import is not sorted into the others.
sys.path.insert(0, os.path.abspath(os.path.join(os.curdir,"..")))
import deviceaccess as da
# fmt: on


class TestAccessors(unittest.TestCase):

    def testScalar(self):

        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        dev = da.Device("CARD_WITH_MODULES")
        dev.open()

        acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        reference = [99]
        acc.set(reference)
        acc.write()

        otherAcc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        otherAcc.read()

        self.assertTrue(reference == otherAcc.view())

        boolAcc = dev.getScalarRegisterAccessor(bool, "ADC/WORD_CLK_CNT_1")
        boolAcc.read()
        self.assertTrue(boolAcc)

        reference = 0
        acc.set(reference)
        acc.write()
        boolAcc.read()
        self.assertFalse(boolAcc)

        boolAcc.set(True)
        boolAcc.write()
        acc.read()
        self.assertTrue(acc == 1)

        dev.close()

    def testOneD(self):

        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        dev = da.Device("CARD_WITH_MODULES")
        dev.open()

        acc = dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        elements = acc.getNElements()
        self.assertTrue(elements == 4)

        reference = [i + 42 for i in range(elements)]
        acc.set(reference)
        acc.write()

        otherAcc = dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        otherAcc.read()
        self.assertTrue(np.array_equal(reference, otherAcc.view()))

        boolAcc = dev.getOneDRegisterAccessor(bool, "BOARD/WORD_CLK_MUX")
        referenceBool = [True]*elements
        boolAcc.read()
        self.assertTrue(np.array_equal(referenceBool, boolAcc.view()))

        reference = [0, 1, 0, 1]
        referenceBool = [False, True, False, True]
        acc.set(reference)
        acc.write()

        boolAcc.read()
        self.assertTrue(np.array_equal(referenceBool, boolAcc.view()))

        referenceBool = [False, False, True, True]
        reference = [0, 0, 1, 1]
        boolAcc.set(referenceBool)
        boolAcc.write()

        acc.read()
        self.assertTrue(np.array_equal(reference, acc.view()))

        dev.close()

    def testTwoD(self):

        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        dev = da.Device("CARD_WITH_MODULES")
        dev.open()
        acc = dev.getTwoDRegisterAccessor(np.int32, "BOARD/DMA")
        channels = acc.getNChannels()
        elementsPerChannel = acc.getNElementsPerChannel()
        reference = [
            [i * j + i + j + 12 for j in range(elementsPerChannel)] for i in range(channels)]

        acc.set(reference)
        acc.write()

        otherAcc = dev.getTwoDRegisterAccessor(np.int32, "BOARD/DMA")
        otherAcc.read()
        self.assertTrue(np.array_equal(reference, otherAcc))

        boolAcc = dev.getTwoDRegisterAccessor(bool, "BOARD/DMA")
        referenceBool = [[True]*elementsPerChannel]*channels
        boolAcc.read()
        self.assertTrue(np.array_equal(referenceBool, boolAcc))

        reference = [
            [1 if i == j else 0 for j in range(elementsPerChannel)] for i in range(channels)]
        referenceBool = [
            [i == j for j in range(elementsPerChannel)] for i in range(channels)]
        acc.set(reference)
        acc.write()

        boolAcc.read()
        self.assertTrue(np.array_equal(referenceBool, boolAcc))

        reference = [
            [1 if i < j else 0 for j in range(elementsPerChannel)] for i in range(channels)]
        referenceBool = [
            [i < j for j in range(elementsPerChannel)] for i in range(channels)]
        boolAcc.set(referenceBool)
        boolAcc.write()

        acc.read()
        self.assertTrue(np.array_equal(reference, acc))

        dev.close()

    def testTypes(self):
        # TODO test all the different userTypes from the Cpp lib
        pass


if __name__ == '__main__':
    unittest.main()

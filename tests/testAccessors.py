#!/usr/bin/env python3
# SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
# SPDX-License-Identifier: LGPL-3.0-or-later

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
        otherAcc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")

        # write through one accessor, read through the other
        reference = [99]
        acc.set(reference)
        acc.write()
        otherAcc.read()
        self.assertTrue(reference == otherAcc)

        # test bool data type
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

        # test writeDestructively
        reference = 42
        acc.set(reference)
        acc.writeDestructively()
        otherAcc.read()
        self.assertTrue(reference == otherAcc)

        # test readNonBlocking
        reference = 120
        acc.set(reference)
        acc.write()
        otherAcc.readNonBlocking()
        self.assertTrue(reference == otherAcc)

        # test readLatest
        reference = 240
        acc.set(reference)
        acc.write()
        otherAcc.readLatest()
        self.assertTrue(reference == otherAcc)

        # test string data type
        stringAcc = dev.getScalarRegisterAccessor(str, "ADC/WORD_CLK_CNT_1")
        stringAcc.read()
        self.assertTrue(str(reference) == stringAcc)

        stringAcc.set('123456789')
        stringAcc.write()
        acc.read()
        self.assertTrue(acc == 123456789)

        # test that setting the string accessor to a longer string than currently works
        stringTooShortAcc = dev.getScalarRegisterAccessor(str, "ADC/WORD_CLK_CNT_1")
        stringTooShortAcc.read()

        self.assertTrue(stringTooShortAcc == '123456789')

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
        self.assertTrue(np.array_equal(reference, otherAcc))
        for i in range(0, elements):
            self.assertTrue(reference[i] == otherAcc[i])

        boolAcc = dev.getOneDRegisterAccessor(bool, "BOARD/WORD_CLK_MUX")
        referenceBool = [True]*elements
        boolAcc.read()
        self.assertTrue(np.array_equal(referenceBool, boolAcc))

        stringAcc = dev.getOneDRegisterAccessor(str, "BOARD/WORD_CLK_MUX")
        stringAcc.read()
        stringReference = [str(i) for i in reference]
        self.assertTrue(np.array_equal(stringReference, stringAcc.get()))

        reference = [0, 1, 0, 1]
        referenceBool = [False, True, False, True]
        acc.set(reference)
        acc.write()

        boolAcc.read()
        self.assertTrue(np.array_equal(referenceBool, boolAcc))

        referenceBool = [False, False, True, True]
        reference = [0, 0, 1, 1]
        boolAcc.set(referenceBool)
        boolAcc.write()

        acc.read()
        self.assertTrue(np.array_equal(reference, acc))

        reference = [i + 54321 for i in range(elements)]
        stringReference = [str(i) for i in reference]
        stringAcc.set(stringReference)
        stringAcc.write()
        acc.read()
        self.assertTrue(np.array_equal(reference, acc))

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

        '''
        # String 2D accessors currently crash in DeviceAccess, see redmine bug #12684
        stringAcc = dev.getTwoDRegisterAccessor(str, "BOARD/DMA")
        stringAcc.read()
        stringReference = [str(i) for i in reference]
        self.assertTrue(np.array_equal(stringReference, stringAcc))
        '''

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

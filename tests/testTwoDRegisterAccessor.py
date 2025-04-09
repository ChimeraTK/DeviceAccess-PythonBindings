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


class TestTwoDRegisterAccessor(unittest.TestCase):

    def testTwoD(self):

        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        dev = da.Device("CARD_WITH_MODULES")
        dev.open()

        acc = dev.getTwoDRegisterAccessor(np.int32, "BOARD/DMA")
        channels = acc.getNChannels()
        elementsPerChannel = acc.getNElementsPerChannel()

        self.assertTrue(acc.getName() == "/BOARD/DMA")

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

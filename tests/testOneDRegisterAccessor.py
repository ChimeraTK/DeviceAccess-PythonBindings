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


class TestOneDRegisterAccessor(unittest.TestCase):

    def testOneD(self):

        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        dev = da.Device("CARD_WITH_MODULES")
        dev.open()

        acc = dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        elements = acc.getNElements()
        self.assertTrue(elements == 4)

        self.assertTrue(acc.getName() == "/BOARD/WORD_CLK_MUX")
        self.assertTrue(acc.getValueType() == np.int32)

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


if __name__ == '__main__':
    unittest.main()

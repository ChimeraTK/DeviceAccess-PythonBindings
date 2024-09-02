#!/usr/bin/env python3
# SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
# SPDX-License-Identifier: LGPL-3.0-or-later

from concurrent.futures import thread
import sys
import unittest
from pathlib import Path
import numpy as np
import os
import threading
from time import sleep


# fmt: off
# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module. Formatting is switched off,
# so the import is not sorted into the others.
sys.path.insert(0, os.path.abspath(os.path.join(os.curdir,"..")))
import deviceaccess as da
# fmt: on

DEVINFO_DIR = Path(__file__).parent / "deviceInformation"

class TestDeviceReadWrite(unittest.TestCase):

    def setUp(self):
        da.setDMapFilePath(str(DEVINFO_DIR / "testCrate.dmap"))
        self.dev = da.Device("TEST_CARD")
        self.dev.open()

        self.twoD_in32_acc = self.dev.getTwoDRegisterAccessor(
            np.int32, "INT32_TEST/2DARRAY")
        self.oneD_int32_acc = self.dev.getOneDRegisterAccessor(
            np.int32, "INT32_TEST/1DARRAY")
        self.scalar_int32_acc = self.dev.getScalarRegisterAccessor(
            np.int32, "INT32_TEST/SCALAR")

    def tearDown(self) -> None:
        self.dev.close()

    def testDeviceRead(self):
        scalarTestValue = 14
        self.scalar_int32_acc.set(scalarTestValue)
        self.scalar_int32_acc.write()

        self.assertEqual(self.dev.read(
            "INT32_TEST/SCALAR", np.int32), scalarTestValue)

        oneDTestArray = [1, 2, 9, 18]
        self.oneD_int32_acc.set(oneDTestArray)
        self.oneD_int32_acc.write()
        for pair in zip(self.dev.read("INT32_TEST/1DARRAY", np.int32), oneDTestArray):
            self.assertEqual(pair[0], pair[1])

    def testDeviceReadWithVariableSize(self):
        oneDTestArray = [1, 2, 9, 18]
        self.oneD_int32_acc.set(oneDTestArray)
        self.oneD_int32_acc.write()
        # test numberOfElements
        for pair in zip(self.dev.read("INT32_TEST/1DARRAY", np.int32, 2), oneDTestArray[0:2], ):
            self.assertEqual(pair[0], pair[1])
        # test invalid numberOfElements
        with self.assertRaises(RuntimeError):  # Should be logic_error, but boost is not forwarding it as expected
            self.dev.read("INT32_TEST/1DARRAY", np.int32, 2765)
        # test offset
        for pair in zip(self.dev.read("INT32_TEST/1DARRAY", np.int32, 2, 2), oneDTestArray[2:4], ):
            self.assertEqual(pair[0], pair[1])
        # test invalid offset
        with self.assertRaises(RuntimeError):  # Should be logic_error, but boost is not forwarding it as expected
            self.dev.read("INT32_TEST/1DARRAY", np.int32, 2, 765)

    def testDeviceWrite(self):
        scalarTestValue = 145
        self.dev.write("INT32_TEST/SCALAR", scalarTestValue)
        self.scalar_int32_acc.read()

        self.assertEqual(self.scalar_int32_acc, scalarTestValue)

        oneDTestArray = [10, 25, 99, -8]
        self.dev.write("INT32_TEST/1DARRAY", oneDTestArray)
        self.oneD_int32_acc.read()

        for pair in zip(self.oneD_int32_acc, oneDTestArray):
            self.assertEqual(pair[0], pair[1])


if __name__ == '__main__':
    unittest.main()

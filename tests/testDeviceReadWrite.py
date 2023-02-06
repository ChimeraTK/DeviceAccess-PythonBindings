#!/usr/bin/env python3
from concurrent.futures import thread
import sys
import unittest
import numpy as np
import os
import threading
from time import sleep


# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module.
sys.path.insert(0, os.path.abspath(os.curdir))
import deviceaccess as da


class TestDeviceReadWrite(unittest.TestCase):

    def setUp(self):
        da.setDMapFilePath("deviceInformation/testCrate.dmap")
        self.dev = da.Device("TEST_CARD")
        self.dev.open()

        self.twoD_in32_acc = self.dev.getTwoDRegisterAccessor(np.int32, "INT32_TEST/2DARRAY")
        self.oneD_int32_acc = self.dev.getOneDRegisterAccessor(np.int32, "INT32_TEST/1DARRAY")
        self.scalar_int32_acc = self.dev.getScalarRegisterAccessor(np.int32, "INT32_TEST/SCALAR")

    def tearDown(self) -> None:
        self.dev.close()

    def testCorrectRead(self):
        scalarTestValue = 14
        self.scalar_int32_acc.set(scalarTestValue)
        self.scalar_int32_acc.write()

        self.assertEqual(self.dev.read("INT32_TEST/SCALAR", np.int32), scalarTestValue)

        oneDTestArray = [1, 2, 9, 18]
        self.oneD_int32_acc.set(oneDTestArray)
        self.oneD_int32_acc.write()
        for pair in zip(self.dev.read("INT32_TEST/1DARRAY", np.int32), oneDTestArray):
            self.assertEqual(pair[0], pair[1])

    def testCorrectWrite(self):
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

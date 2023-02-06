#!/usr/bin/env python3
import deviceaccess as da
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


class TestConvenienceFunctions(unittest.TestCase):

    def setUp(self):
        da.setDMapFilePath("deviceInformation/testCrate.dmap")
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

    def testReadAndGet(self):
        scalarTestValue = 14
        self.scalar_int32_acc.set(scalarTestValue)
        self.scalar_int32_acc.write()
        result_from_readAndGet = self.scalar_int32_acc.readAndGet()

        self.assertEqual(result_from_readAndGet, scalarTestValue)

    def testSetAndWrite(self):
        scalarTestValue = 145
        self.scalar_int32_acc.setAndWrite(scalarTestValue)
        self.scalar_int32_acc.read()
        actual_register = self.scalar_int32_acc

        self.assertEqual(actual_register, scalarTestValue)

    def testWriteIfDifferent(self):
        scalarTestValue = 232
        vn_first = da.VersionNumber()
        self.scalar_int32_acc.writeIfDifferent(scalarTestValue, vn_first)
        vn_second = da.VersionNumber()
        self.scalar_int32_acc.writeIfDifferent(scalarTestValue, vn_second)
        vn_after = self.scalar_int32_acc.getVersionNumber()
        self.assertEqual(vn_first, vn_after)
        self.assertGreater(vn_second, vn_after)


if __name__ == '__main__':
    unittest.main()

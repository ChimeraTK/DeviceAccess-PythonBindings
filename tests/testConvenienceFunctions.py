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

        scalarTestValue = 14
        first_version_number = da.VersionNumber()
        second_version_number = da.VersionNumber()
        self.scalar_int32_acc.setAndWrite(scalarTestValue, second_version_number)
        self.scalar_int32_acc.read()
        actual_register = self.scalar_int32_acc

        self.assertEqual(actual_register, scalarTestValue)
        with self.assertRaises(RuntimeError):
            self.scalar_int32_acc.setAndWrite(scalarTestValue, first_version_number)

    def testWriteIfDifferent(self):
        # Test if values are correctly written
        scalarTestValue = 232
        vn_first = da.VersionNumber()
        vn_second = da.VersionNumber()
        self.scalar_int32_acc.writeIfDifferent(scalarTestValue, vn_first)
        self.scalar_int32_acc.read()
        self.assertEqual(scalarTestValue, self.scalar_int32_acc)
        self.scalar_int32_acc.writeIfDifferent(scalarTestValue, vn_second)
        self.scalar_int32_acc.read()
        self.assertEqual(scalarTestValue, self.scalar_int32_acc)

        scalarTestValue = 99
        self.scalar_int32_acc.writeIfDifferent(scalarTestValue)
        self.scalar_int32_acc.read()
        self.assertEqual(scalarTestValue, self.scalar_int32_acc)
        self.scalar_int32_acc.writeIfDifferent(scalarTestValue)
        self.scalar_int32_acc.read()
        self.assertEqual(scalarTestValue, self.scalar_int32_acc)

        # assert that version number stays the same as an indicator, that nothing has been written
        scalarTestValue = 232
        vn_first = da.VersionNumber()
        self.scalar_int32_acc.writeIfDifferent(scalarTestValue, vn_first)
        vn_second = da.VersionNumber()
        self.scalar_int32_acc.writeIfDifferent(scalarTestValue, vn_second)
        vn_after = self.scalar_int32_acc.getVersionNumber()
        self.assertTrue(vn_first == vn_after)
        self.assertGreater(vn_second, vn_after)

        scalarTestValue = 99
        self.scalar_int32_acc.writeIfDifferent(scalarTestValue)
        vn_after_first_write_attemp = self.scalar_int32_acc.getVersionNumber()
        self.scalar_int32_acc.writeIfDifferent(scalarTestValue)
        vn_after_second_write_attemp = self.scalar_int32_acc.getVersionNumber()
        self.assertEqual(vn_after_first_write_attemp, vn_after_second_write_attemp)


if __name__ == '__main__':
    unittest.main()

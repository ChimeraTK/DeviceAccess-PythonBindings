#!/usr/bin/env python3
from concurrent.futures import thread
import sys
import unittest
import numpy as np
import os
import threading
from time import sleep


# fmt: off
# This is a hack for now. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module. Formatting is switched off,
# so the import is not sorted into the others.
sys.path.insert(0, os.path.abspath(os.curdir))
import deviceaccess as da
import _da_python_bindings as pb
# fmt: on


class TestDeviceAccessLib(unittest.TestCase):

    def setUp(self):
        self.dmap_filepath = "deviceInformation/testCrate.dmap"
        da.setDMapFilePath(self.dmap_filepath)
        self.dev = da.Device("TEST_CARD")
        self.dev.open()
        self.dev.activateAsyncRead()
        self.int_2d_int32_acc_name = "/INT32_TEST/2DARRAY"
        self.twoD_in32_acc = self.dev.getTwoDRegisterAccessor(
            np.int32, self.int_2d_int32_acc_name)
        self.oneD_int32_acc = self.dev.getOneDRegisterAccessor(
            np.int32, "/INT32_TEST/1DARRAY")
        self.scalar_int32_acc = self.dev.getScalarRegisterAccessor(
            np.int32, "/INT32_TEST/SCALAR")

        self.async_read_1d_int32_acc = self.dev.getOneDRegisterAccessor(
            np.int32, "/ASYNC/TEST_AREA_PUSH", accessModeFlags=[da.AccessMode.wait_for_new_data])

        self.raw_1d_int32_acc = self.dev.getOneDRegisterAccessor(
            np.int32, "/ASYNC/TEST_AREA_PUSH", accessModeFlags=[da.AccessMode.raw])

        self.async_raw_1d_int32_acc = self.dev.getOneDRegisterAccessor(
            np.int32, "/ASYNC/TEST_AREA_PUSH", accessModeFlags=[da.AccessMode.raw, da.AccessMode.wait_for_new_data])

        self.scalar_data_ready = self.dev.getVoidRegisterAccessor(
            "DUMMY_INTERRUPT_2")
        self.write_scalar = self.dev.getScalarRegisterAccessor(
            np.int32, "/ASYNC/SCALAR")
        self.async_scalar_push = self.dev.getScalarRegisterAccessor(
            np.int32, "/ASYNC/SCALAR_PUSH", accessModeFlags=[da.AccessMode.wait_for_new_data])

        self.rw_int32_acc = self.dev.getScalarRegisterAccessor(
            np.int32, "/RW_TEST/SCALAR")
        self.ro_int32_acc = self.dev.getScalarRegisterAccessor(
            np.int32, "/RO_TEST/SCALAR")
        self.wo_int32_acc = self.dev.getScalarRegisterAccessor(
            np.int32, "/WO_TEST/SCALAR")

    def tearDown(self) -> None:
        self.dev.close()

    def testReads_DMAP_file_path(self):
        readout = da.getDMapFilePath()
        self.assertEqual(readout, self.dmap_filepath)

    def testReads_Register_Name(self):
        readout = self.twoD_in32_acc.getName()

        self.assertEqual(readout, self.int_2d_int32_acc_name)

    def testRead_Unit_from_Acc(self):
        readout_unset = self.twoD_in32_acc.getUnit()  # unit is not set
        self.assertEqual(readout_unset, "n./a.")
        # Other values cannot be set. Has to be set via custom backend to be tested

    def testCheck_Access_Mode_Flags(self):
        # standard flags are empty:
        empty_flags = self.twoD_in32_acc.getAccessModeFlags()
        self.assertEqual([], empty_flags)

        # wait_for_new_data:
        wait_for_new_data_flag = self.async_read_1d_int32_acc.getAccessModeFlags()
        self.assertEqual([da.AccessMode.wait_for_new_data], wait_for_new_data_flag)

        # check raw
        raw_data_flag = self.raw_1d_int32_acc.getAccessModeFlags()
        self.assertEqual([da.AccessMode.raw], raw_data_flag)

        # check both
        raw_and_async_data_flags = self.async_raw_1d_int32_acc.getAccessModeFlags()
        self.assertIn(da.AccessMode.raw, raw_and_async_data_flags)
        self.assertIn(da.AccessMode.wait_for_new_data, raw_and_async_data_flags)

    def testReadWriteFlags(self):
        self.assertTrue(self.rw_int32_acc.isReadable())
        self.assertTrue(self.ro_int32_acc.isReadable())
        self.assertFalse(self.wo_int32_acc.isReadable())

        self.assertTrue(self.rw_int32_acc.isWriteable())
        self.assertFalse(self.ro_int32_acc.isWriteable())
        self.assertTrue(self.wo_int32_acc.isWriteable())

        self.assertFalse(self.rw_int32_acc.isReadOnly())
        self.assertTrue(self.ro_int32_acc.isReadOnly())
        self.assertFalse(self.wo_int32_acc.isReadOnly())

    def testReadLatest(self):
        # pay attention to use a normal dummy, shared memory dummy will have a race condition that will fail the test

        # discard first, non-blocking read
        self.async_scalar_push.read()

        # fill the queue with 1 to 10 (10 values):
        for i in range(1, 11):
            self.write_scalar.set(i)
            self.write_scalar.write()
            self.scalar_data_ready.write()

        # queue is full:
        new_value_available = self.async_scalar_push.readLatest()
        self.assertTrue(new_value_available)
        result = self.async_scalar_push
        self.assertEqual(result, 10)

        # queue is empty:
        new_value_available = self.async_scalar_push.readLatest()
        self.assertFalse(new_value_available)

    def testReadNonBlocking(self):
        # pay attention to use a normal dummy, shared memory dummy will have a race condition that will fail the test
        some_value = 445
        some_other_value = -9

        # discard first, non-blocking read
        self.async_scalar_push.read()

        # put one in the queue
        self.write_scalar.set(some_value)
        self.write_scalar.write()
        self.scalar_data_ready.write()
        self.write_scalar.set(some_other_value)
        self.write_scalar.write()
        self.scalar_data_ready.write()

        # queue not empty:
        new_value_available = self.async_scalar_push.readNonBlocking()
        self.assertTrue(new_value_available)
        result = self.async_scalar_push
        self.assertEqual(result, some_value)
        # twice
        new_value_available = self.async_scalar_push.readNonBlocking()
        self.assertTrue(new_value_available)
        result = self.async_scalar_push
        self.assertEqual(result, some_other_value)

        # queue is empty:
        new_value_available = self.async_scalar_push.readNonBlocking()
        self.assertFalse(new_value_available)
        self.assertEqual(result, some_other_value)

    def testWriteDestructivelyDoesWriteAtAll(self):

        scalar_value = 5
        self.scalar_int32_acc.set(scalar_value)
        self.scalar_int32_acc.writeDestructively()
        self.scalar_int32_acc.read()
        self.assertEqual(scalar_value, self.scalar_int32_acc)

        one_d_vector = []
        for i in range(self.oneD_int32_acc.getNElements()):
            one_d_vector.append(2*i + 3)
        self.oneD_int32_acc.set(one_d_vector)
        self.oneD_int32_acc.writeDestructively()
        self.oneD_int32_acc.read()
        for i in range(self.oneD_int32_acc.getNElements()):
            self.assertEqual(self.oneD_int32_acc[i], one_d_vector[i])

        two_d_vector = []
        for i in range(self.twoD_in32_acc.getNChannels()):
            temp = []
            for j in range(self.twoD_in32_acc.getNElementsPerChannel()):
                temp.append(i*2 + j * 3)
            two_d_vector.append(temp)

        self.twoD_in32_acc.set(two_d_vector)
        self.twoD_in32_acc.writeDestructively()
        self.twoD_in32_acc.read()
        for i in range(self.twoD_in32_acc.getNChannels()):
            for j in range(self.twoD_in32_acc.getNElementsPerChannel()):
                self.assertEqual(self.twoD_in32_acc[i][j], two_d_vector[i][j])

    def testValueTypeReturns(self):
        # "types" are defined by getting them from the device
        types_to_tests = [np.int8, np.int32, np.int64, np.float32, np.float64, np.uint8]
        for valuetype in types_to_tests:
            acc = self.dev.getScalarRegisterAccessor(valuetype, "/INT32_TEST/SCALAR")
            self.assertEqual(valuetype, acc.getValueType())  # reported from lib
            self.assertEqual(valuetype, acc.dtype)  # actual numpy type

    def testInitialisation(self):
        self.assertTrue(self.twoD_in32_acc.isInitialised())
        # As the register path is a required argument in this module's implementation
        # it is not possible to tests an uninitialised accessor.

    def testDataValidity(self):
        self.scalar_int32_acc.set(5)
        self.scalar_int32_acc.write()
        self.assertEqual(self.scalar_int32_acc.dataValidity(), da.DataValidity.ok)
        self.scalar_int32_acc.set(6)
        self.scalar_int32_acc.write()
        self.scalar_int32_acc.setDataValidity(da.DataValidity.faulty)
        self.assertEqual(self.scalar_int32_acc.dataValidity(), da.DataValidity.faulty)
        self.scalar_int32_acc.set(7)
        self.scalar_int32_acc.write()
        self.scalar_int32_acc.setDataValidity(da.DataValidity.ok)
        self.assertEqual(self.scalar_int32_acc.dataValidity(), da.DataValidity.ok)

    def testAccessorIDs(self):
        scalar1 = self.dev.getScalarRegisterAccessor(np.int32, "/INT32_TEST/SCALAR")
        scalar2 = self.dev.getScalarRegisterAccessor(np.int32, "/INT32_TEST/SCALAR")
        copy_of_scalar1 = scalar1

        self.assertEqual(scalar1.getId(), copy_of_scalar1.getId())
        self.assertNotEqual(scalar1.getId(), scalar2.getId())

    def testVoidAccessorFunctions(self):
        write_void = self.dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_2")
        read_void = self.dev.getVoidRegisterAccessor(
            "/ASYNC/SCALAR_PUSH", accessModeFlags=[da.AccessMode.wait_for_new_data])

        self.assertFalse(write_void.writeDestructively())

        self.assertTrue(read_void.readLatest())
        self.assertFalse(read_void.readNonBlocking())

        write_void.write()

        self.assertTrue(read_void.readNonBlocking())
        self.assertFalse(read_void.readLatest())

        with self.assertRaises(RuntimeError):
            write_void.read()
        with self.assertRaises(RuntimeError):
            write_void.readLatest()
        with self.assertRaises(RuntimeError):
            write_void.readNonBlocking()

        with self.assertRaises(RuntimeError):
            read_void.write()
        with self.assertRaises(RuntimeError):
            read_void.writeDestructively()


if __name__ == '__main__':
    unittest.main()

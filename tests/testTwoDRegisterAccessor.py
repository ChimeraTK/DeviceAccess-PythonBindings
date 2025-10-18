#!/usr/bin/env python3
# SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
# SPDX-License-Identifier: LGPL-3.0-or-later

import sys
import threading
import time
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

########################################################################################################################

types_to_test = [np.int8, np.uint8, np.int16, np.uint16, np.int32,
                 np.uint32, np.int64, np.uint64, np.float32, np.float64, bool, str]


# Triplet of operator, useForFloat, useForBool, useForStr
# Note: This list is not complete, since some of the operators are a bit harder to test in a generic way, but all
# operators are treated identically anyway.
binaryOps = [
    ("__lt__", True, True, True),
    ("__le__", True, True, True),
    ("__gt__", True, True, True),
    ("__ge__", True, True, True),
    ("__eq__", True, True, True),
    ("__ne__", True, True, True),
    ("__add__", True, True, False),
    ("__iadd__", True, True, False),
    ("__sub__", True, False, False),
    ("__isub__", True, False, False),
    ("__mul__", True, True, False),
    ("__imul__", True, True, False),
    ("__truediv__", True, False, False),
    ("__floordiv__", True, False, False),
    ("__mod__", True, False, False),
    ("__pow__", True, True, False),
    ("__rshift__", False, True, False),
    ("__irshift__", False, False, False),
    ("__ilshift__", False, False, False),
    ("__and__", False, True, False),
    ("__iand__", False, True, False),
    ("__or__", False, True, False),
    ("__ior__", False, True, False),
    ("__xor__", False, True, False),
    ("__ixor__", False, True, False),
]

unaryOps = [
    ("__str__", True, True, True),
    ("__bool__", True, True, True),
]

# Start value used to generate numbers in the value() function below.
generator_seed = 10

########################################################################################################################


def valueAfterConstruct(type):
    """
    Return value-after-construct for the given type, with a length of 4 elements (matching BOARD.WORD_CLK_MUX)
    """
    if type == str:
        return [["", "", "", "", "", ""], ["", "", "", "", "", ""], ["", "", "", "", "", ""], ["", "", "", "", "", ""]]
    if type == bool:
        return [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]
    return [[type(0), type(0), type(0), type(0), type(0), type(0)], [type(0), type(0), type(0), type(0), type(0), type(0)], [type(0), type(0), type(0), type(0), type(0), type(0)], [type(0), type(0), type(0), type(0), type(0), type(0)]]

########################################################################################################################


def value(type, forceUnequal=None):
    """
    Generate a value of the given type, which differs from the value "forceUnequal". If "forceUnequal" is None,
    the generated value will be different from the value-after-construct.
    """
    global generator_seed

    while True:
        generator_seed += 1
        if type == str:
            value = [[str(generator_seed), "42", "120", "17", "23", "68"], [
                "1", "2", "3", "4", "5", "6"], ["7", "8", "9", "10", "11", "12"], ["13", "14", "15", "16", "17", "18"]]
        elif type == bool:
            value = [[(generator_seed % 2 == 0), False, False, True, True, False],
                     [True, False, False, False, False, False], [True, True, False, False, False, False], [True, True, True, False, False, False]]
        else:
            # don't need to cover the full range, so we treat all other's equal
            value = [[type(generator_seed % 100), type(42), type(
                12), type(17), type(23), type(24)], [1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16], [21, 22, 23, 24, 25, 26]]

        if forceUnequal is not None:
            useValue = (value != forceUnequal).any()
        else:
            useValue = True

        if useValue:
            return np.array(value).astype(type)

        generator_seed += 1

########################################################################################################################


def catchEx(lambdaExpression):
    """
    Execute the given lambdaExpression and return its return value. If a ValueError or IndexError is thrown, catch
    and return it.
    """
    try:
        return lambdaExpression()
    except ValueError as ex:
        return ('ValueError', str(ex))
    except IndexError as ex:
        return ('IndexError', str(ex))

########################################################################################################################
########################################################################################################################


class TestTwoDRegisterAccessor(unittest.TestCase):

    def setUp(self):
        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        self.dev = da.Device("CARD_WITH_MODULES")
        self.dev.open()
        self.dev.activateAsyncRead()

        # The "backdoor" accessor is used to check the result of write operations resp. provide data for read
        # operations. This kind of creates a "circular reasoning" for the test, which shall be deemed to be acceptable,
        # since the code under test is merely delegating to well tested C++ code of DeviceAccess.
        self.backdoor = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA")
        self.interrupt = self.dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_0")

    def testGetSet(self):
        for type in types_to_test:
            with self.subTest(type=type):
                acc = self.dev.getTwoDRegisterAccessor(type, "BOARD.DMA")

                self.assertTrue((acc.get() == valueAfterConstruct(type)).all())

                expected = value(type)
                acc.set(expected)

                self.assertTrue((acc.get() == expected).all())

                expected = value(type, expected)
                acc.set(expected)

                self.assertTrue((acc.get() == expected).all())

    def testRead(self):
        for type in types_to_test:
            with self.subTest(type=type):
                if type == str:
                    print('========= SKIPPING TEST as it would fail!', flush=True)
                    continue

                acc = self.dev.getTwoDRegisterAccessor(type, "BOARD.DMA")

                expected = value(type)
                self.backdoor.set(expected)
                self.backdoor.write()

                acc.read()

                self.assertTrue((acc == expected).all())

    def testRead_push(self):
        for type in types_to_test:
            with self.subTest(type=type):
                if type == str:
                    print('========= SKIPPING TEST as it would fail!', flush=True)
                    continue

                acc = self.dev.getTwoDRegisterAccessor(
                    type, "BOARD.DMA_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])
                acc.read()  # initial value

                t = threading.Thread(name='blocking_read', target=lambda acc=acc: acc.read())
                t.start()

                time.sleep(0.1)
                self.assertTrue(t.is_alive())  # read is not yet complete

                expected = value(type)
                self.backdoor.set(expected)
                self.backdoor.write()

                self.interrupt.write()
                t.join(1)  # TODO increase to 10s
                self.assertFalse(t.is_alive())  # read has completed

                self.assertTrue((acc == expected).all())

    def testReadLatest(self):
        for type in types_to_test:
            with self.subTest(type=type):
                if type == str:
                    print('========= SKIPPING TEST as it would fail!', flush=True)
                    continue

                acc = self.dev.getTwoDRegisterAccessor(
                    type, "BOARD.DMA_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])

                expected = value(type)
                self.backdoor.set(expected)
                self.backdoor.write()
                self.interrupt.write()

                retval = acc.readLatest()

                self.assertTrue((acc == expected).all())
                self.assertTrue(retval)

                retval = acc.readLatest()
                self.assertFalse(retval)

    def testReadNonBlocking(self):
        for type in types_to_test:
            with self.subTest(type=type):
                if type == str:
                    print('========= SKIPPING TEST as it would fail!', flush=True)
                    continue

                expected1 = value(type)
                self.backdoor.set(expected1)
                self.backdoor.write()

                acc = self.dev.getTwoDRegisterAccessor(
                    type, "BOARD.DMA_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])

                expected2 = value(type, expected1)
                self.backdoor.set(expected2)
                self.backdoor.write()
                self.interrupt.write()

                retval = acc.readNonBlocking()

                self.assertTrue((acc == expected1).all())
                self.assertTrue(retval)

                retval = acc.readNonBlocking()

                self.assertTrue((acc == expected2).all())
                self.assertTrue(retval)

                retval = acc.readNonBlocking()
                self.assertFalse(retval)

    def testWrite(self):
        for type in types_to_test:
            with self.subTest(type=type):
                if type == str:
                    print('========= SKIPPING TEST as it would fail!', flush=True)
                    continue

                acc = self.dev.getTwoDRegisterAccessor(type, "BOARD.DMA")

                expected = value(type)
                acc.set(expected)

                acc.write()

                self.backdoor.read()
                self.assertTrue((self.backdoor.astype(type) == expected).all(), f'{self.backdoor} == {expected}')

    def testWriteDestructively(self):
        for type in types_to_test:
            with self.subTest(type=type):
                if type == str:
                    print('========= SKIPPING TEST as it would fail!', flush=True)
                    continue

                acc = self.dev.getTwoDRegisterAccessor(type, "BOARD.DMA")

                expected = value(type)
                acc.set(expected)

                acc.writeDestructively()

                self.backdoor.read()
                self.assertTrue((self.backdoor.astype(type) == expected).all(), f'{self.backdoor} == {expected}')

    def testGetName(self):
        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.WORD_CLK_MUX")
        self.assertTrue(acc.getName() == "/BOARD.WORD_CLK_MUX")

        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.WORD_CLK_MUX_3")
        self.assertTrue(acc.getName() == "/BOARD.WORD_CLK_MUX_3")

    def testGetValueType(self):
        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA")
        self.assertTrue(acc.getValueType() == np.int32)

        acc = self.dev.getTwoDRegisterAccessor(np.float32, "BOARD.DMA")
        self.assertTrue(acc.getValueType() == np.float32)

    def testGetAccessModeFlags(self):
        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA")
        self.assertTrue(acc.getAccessModeFlags() == [])

        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA_INT", accessModeFlags=[
                                               da.AccessMode.wait_for_new_data])
        self.assertTrue(acc.getAccessModeFlags() == [da.AccessMode.wait_for_new_data])

    def testGetVersionNumber(self):
        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA")

        self.assertTrue(acc.getVersionNumber() == da.VersionNumber.getNullVersion())

        myVersion = da.VersionNumber()
        acc.read()

        self.assertTrue(acc.getVersionNumber() != da.VersionNumber.getNullVersion())
        self.assertTrue(acc.getVersionNumber() > myVersion)

    def testIsReadOnly(self):
        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA")
        self.assertTrue(acc.isReadOnly() == False)

        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA_RO")
        self.assertTrue(acc.isReadOnly() == True)

    def testIsReadable(self):
        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA")
        self.assertTrue(acc.isReadable() == True)

        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA_WO")
        self.assertTrue(acc.isReadable() == False)

    def testIsWriteable(self):
        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA")
        self.assertTrue(acc.isWriteable() == True)

        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA_RO")
        self.assertTrue(acc.isWriteable() == False)

    def testIsWriteable(self):
        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA")
        self.assertTrue(acc.isInitialised() == True)
        # TODO: unclear how to get an uninitialised accessor!? Maybe remove this function?

    def testDataValidity(self):
        # The backend used for this test cannot deal with data validity, so we just test setDataValidity() and
        # dataValidity() of a single accessor instance
        acc = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA")
        self.assertTrue(acc.dataValidity() == da.DataValidity.ok)
        acc.setDataValidity(da.DataValidity.faulty)
        self.assertTrue(acc.dataValidity() == da.DataValidity.faulty)
        acc.setDataValidity(da.DataValidity.ok)
        self.assertTrue(acc.dataValidity() == da.DataValidity.ok)

    def testGetId(self):
        acc1 = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA")
        acc2 = self.dev.getTwoDRegisterAccessor(np.int32, "BOARD.DMA")
        acc1_copy = acc1  # testing with a copy barely makes a difference in Python, doesn't hurt anyway
        self.assertTrue(acc1.getId() != acc2.getId())
        self.assertTrue(acc1.getId() == acc1_copy.getId())

    def testInterrupt(self):
        acc = self.dev.getTwoDRegisterAccessor(
            np.int32, "BOARD.DMA_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])
        acc.read()  # initial value

        def myThreadFun(acc):
            try:
                acc.read()
            except RuntimeError:
                pass

        t = threading.Thread(name='blocking_read', target=lambda acc=acc: myThreadFun(acc))
        t.start()

        time.sleep(0.1)
        self.assertTrue(t.is_alive())  # read is not yet complete

        acc.interrupt()

        t.join(10)
        self.assertFalse(t.is_alive())  # read has completed

    def testBinaryOperators(self):
        for type in types_to_test:
            # registers don't matter since we do not actually execute any transfer operations
            acc1 = self.dev.getTwoDRegisterAccessor(type, "BOARD.DMA")
            acc2 = self.dev.getTwoDRegisterAccessor(type, "BOARD.DMA")
            for operator, useForFloat, useForBool, useForStr in binaryOps:

                if type == str and not useForStr:
                    continue
                if type == bool and not useForBool:
                    continue
                if (type == np.float32) or (type == np.float64) and not useForFloat:
                    continue

                with self.subTest(type=type, operator=operator):
                    val1 = np.array(value(type))
                    val2 = np.array(value(type, val1))
                    acc1.set(val1)
                    acc2.set(val2)

                    expected12 = catchEx(lambda: val1.__getattribute__(operator)(val2))
                    expected21 = catchEx(lambda: val2.__getattribute__(operator)(val1))

                    self.assertTrue((catchEx(lambda: acc1.__getattribute__(operator)(acc2)) == expected12).all())
                    self.assertTrue((catchEx(lambda: acc1.__getattribute__(operator)(val2)) == expected12).all())
                    self.assertTrue((catchEx(lambda: acc2.__getattribute__(operator)(acc1)) == expected21).all())
                    self.assertTrue((catchEx(lambda: acc2.__getattribute__(operator)(val1)) == expected21).all())

                    acc2.set(val1)
                    expected11 = catchEx(lambda: val1.__getattribute__(operator)(val1))

                    self.assertTrue((catchEx(lambda: acc1.__getattribute__(operator)(acc2)) == expected11).all())
                    self.assertTrue((catchEx(lambda: acc1.__getattribute__(operator)(val1)) == expected11).all())

    def testUnaryOperators(self):
        for type in types_to_test:
            # registers don't matter since we do not actually execute any transfer operations
            acc = self.dev.getTwoDRegisterAccessor(type, "BOARD.DMA")
            for operator, useForFloat, useForBool, useForStr in unaryOps:

                if type == str and not useForStr:
                    continue
                if type == bool and not useForBool:
                    continue
                if (type == np.float32) or (type == np.float64) and not useForFloat:
                    continue

                with self.subTest(type=type, operator=operator):
                    for i in range(0, 2):

                        val = np.array(value(type))
                        acc.set(val)

                        expected = catchEx(lambda: val.__getattribute__(operator)())

                        self.assertEqual(catchEx(lambda: acc.__getattribute__(operator)()), expected)

########################################################################################################################


if __name__ == '__main__':
    unittest.main()

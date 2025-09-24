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

#####################################################################################################################

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
    ("__bool__", True, True, False),
]

# Start value used to generate numbers in the value() function below.
generator_seed = 10

#####################################################################################################################


def valueAfterConstruct(type):
    """
    Return value-after-construct for the given type, with a length of 4 elements (matching BOARD.WORD_CLK_MUX)
    """
    if type == str:
        return ["", "", "", ""]
    if type == bool:
        return [False, False, False, False]
    return [type(0), type(0), type(0), type(0)]

#####################################################################################################################


def value(type, forceUnequal=None):
    """
    Generate a value of the given type, which differs from the value "forceUnequal". If "forceUnequal" is None,
    the generated value will be different from the value-after-construct.
    """
    global generator_seed

    while True:
        generator_seed += 1
        if type == str:
            value = [str(generator_seed), "42", "120", "17"]
        elif type == bool:
            value = [(generator_seed % 2 == 0), False, False, True]
        else:
            # don't need to cover the full range, so we treat all other's equal
            value = [type(generator_seed % 100), type(42), type(120), type(17)]

        if forceUnequal is not None:
            useValue = value != list(forceUnequal)
        else:
            useValue = True

        if useValue:
            if type != str:
                return np.array(value)
            else:
                return value

        generator_seed += 1

#####################################################################################################################


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

#####################################################################################################################
#####################################################################################################################


class TestOneDRegisterAccessor(unittest.TestCase):

    def setUp(self):
        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        self.dev = da.Device("CARD_WITH_MODULES")
        self.dev.open()
        self.dev.activateAsyncRead()

        # The "backdoor" accessor is used to check the result of write operations resp. provide data for read
        # operations. This kind of creates a "circular reasoning" for the test, which shall be deemed to be acceptable,
        # since the code under test is merely delegating to well tested C++ code of DeviceAccess.
        self.backdoor = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        self.interrupt = self.dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_0")

    def testGetSet(self):
        for typ in types_to_test:
            with self.subTest(type=typ):
                acc = self.dev.getOneDRegisterAccessor(typ, "BOARD/WORD_CLK_MUX")

                if typ != str:
                    self.assertTrue((acc.get() == valueAfterConstruct(typ)).all())
                else:
                    self.assertTrue(acc.get() == valueAfterConstruct(typ))

                expected = value(typ)
                acc.set(expected)

                if typ != str:
                    self.assertTrue((acc.get() == expected).all())
                else:
                    self.assertTrue(acc.get() == expected)

                expected = value(typ, expected)
                acc.set(expected)

                if typ != str:
                    self.assertTrue((acc.get() == expected).all())
                else:
                    self.assertTrue(acc.get() == expected)

    def testRead(self):
        for typ in types_to_test:
            with self.subTest(type=typ):
                acc = self.dev.getOneDRegisterAccessor(typ, "BOARD/WORD_CLK_MUX")

                expected = value(typ)
                self.backdoor.set(expected)
                self.backdoor.write()

                acc.read()

                if typ != str:
                    self.assertTrue((acc == expected).all())
                else:
                    self.assertTrue(acc == expected)

    def testRead_push(self):
        for typ in types_to_test:
            with self.subTest(type=typ):
                acc = self.dev.getOneDRegisterAccessor(
                    typ, "BOARD/WORD_CLK_MUX_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])
                acc.read()  # initial value

                t = threading.Thread(name='blocking_read', target=lambda acc=acc: acc.read())
                t.start()

                time.sleep(0.1)
                self.assertTrue(t.is_alive())  # read is not yet complete

                expected = value(typ)
                self.backdoor.set(expected)
                self.backdoor.write()

                self.interrupt.write()
                t.join(1)  # TODO increase to 10s
                self.assertFalse(t.is_alive())  # read has completed

                if typ != str:
                    self.assertTrue((acc == expected).all())
                else:
                    self.assertTrue(acc == expected)

    def testReadLatest(self):
        for typ in types_to_test:
            with self.subTest(type=typ):
                acc = self.dev.getOneDRegisterAccessor(
                    typ, "BOARD/WORD_CLK_MUX_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])

                expected = value(typ)
                self.backdoor.set(expected)
                self.backdoor.write()
                self.interrupt.write()

                retval = acc.readLatest()

                if typ != str:
                    self.assertTrue((acc == expected).all())
                else:
                    self.assertTrue(acc == expected)
                self.assertTrue(retval)

                retval = acc.readLatest()
                self.assertFalse(retval)

    def testReadNonBlocking(self):
        for typ in types_to_test:
            with self.subTest(type=typ):
                expected1 = value(typ)
                self.backdoor.set(expected1)
                self.backdoor.write()

                acc = self.dev.getOneDRegisterAccessor(
                    typ, "BOARD/WORD_CLK_MUX_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])

                expected2 = value(typ, expected1)
                self.backdoor.set(expected2)
                self.backdoor.write()
                self.interrupt.write()

                retval = acc.readNonBlocking()

                if typ != str:
                    self.assertTrue((acc == expected1).all())
                else:
                    self.assertTrue(acc == expected1)
                self.assertTrue(retval)

                retval = acc.readNonBlocking()

                if typ != str:
                    self.assertTrue((acc == expected2).all())
                else:
                    self.assertTrue(acc == expected2)
                self.assertTrue(retval)

                retval = acc.readNonBlocking()
                self.assertFalse(retval)

    def testWrite(self):
        for typ in types_to_test:
            with self.subTest(type=typ):
                acc = self.dev.getOneDRegisterAccessor(typ, "BOARD/WORD_CLK_MUX")

                expected = value(typ)
                acc.set(expected)

                acc.write()

                self.backdoor.read()
                self.assertTrue((self.backdoor.astype(typ) == expected).all(), f'{self.backdoor} == {expected}')

    def testWriteDestructively(self):
        for typ in types_to_test:
            with self.subTest(type=typ):
                acc = self.dev.getOneDRegisterAccessor(typ, "BOARD/WORD_CLK_MUX")

                expected = value(typ)
                acc.set(expected)

                acc.writeDestructively()

                self.backdoor.read()
                self.assertTrue((self.backdoor.astype(typ) == expected).all(), f'{self.backdoor} == {expected}')

    def testGetName(self):
        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD.WORD_CLK_MUX")
        self.assertTrue(acc.getName() == "/BOARD.WORD_CLK_MUX")

        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD.WORD_CLK_MUX_3")
        self.assertTrue(acc.getName() == "/BOARD.WORD_CLK_MUX_3")

    def testGetValueType(self):
        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        self.assertTrue(acc.getValueType() == np.int32)

        acc = self.dev.getOneDRegisterAccessor(np.float32, "BOARD/WORD_CLK_MUX")
        self.assertTrue(acc.getValueType() == np.float32)

    def testGetAccessModeFlags(self):
        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        self.assertTrue(acc.getAccessModeFlags() == [])

        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX", accessModeFlags=[da.AccessMode.raw])
        self.assertTrue(acc.getAccessModeFlags() == [da.AccessMode.raw])

    def testGetVersionNumber(self):
        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")

        self.assertTrue(acc.getVersionNumber() == da.VersionNumber.getNullVersion())

        myVersion = da.VersionNumber()
        acc.read()

        self.assertTrue(acc.getVersionNumber() != da.VersionNumber.getNullVersion())
        self.assertTrue(acc.getVersionNumber() > myVersion)

    def testIsReadOnly(self):
        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        self.assertTrue(acc.isReadOnly() == False)

        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX_RO")
        self.assertTrue(acc.isReadOnly() == True)

    def testIsReadable(self):
        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        self.assertTrue(acc.isReadable() == True)

        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX_WO")
        self.assertTrue(acc.isReadable() == False)

    def testIsWriteable(self):
        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        self.assertTrue(acc.isWriteable() == True)

        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX_RO")
        self.assertTrue(acc.isWriteable() == False)

    def testIsWriteable(self):
        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        self.assertTrue(acc.isInitialised() == True)
        # TODO: unclear how to get an uninitialised accessor!? Maybe remove this function?

    def testDataValidity(self):
        # The backend used for this test cannot deal with data validity, so we just test setDataValidity() and
        # dataValidity() of a single accessor instance
        acc = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        self.assertTrue(acc.dataValidity() == da.DataValidity.ok)
        acc.setDataValidity(da.DataValidity.faulty)
        self.assertTrue(acc.dataValidity() == da.DataValidity.faulty)
        acc.setDataValidity(da.DataValidity.ok)
        self.assertTrue(acc.dataValidity() == da.DataValidity.ok)

    def testGetId(self):
        acc1 = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        acc2 = self.dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
        acc1_copy = acc1  # testing with a copy barely makes a difference in Python, doesn't hurt anyway
        self.assertTrue(acc1.getId() != acc2.getId())
        self.assertTrue(acc1.getId() == acc1_copy.getId())

    def testInterrupt(self):
        acc = self.dev.getOneDRegisterAccessor(
            np.int32, "BOARD/WORD_CLK_MUX_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])
        acc.read()  # initial value

        def myThreadFun(acc):
            try:
                acc.read()
            except da.ThreadInterrupted:
                pass

        t = threading.Thread(name='blocking_read', target=lambda acc=acc: myThreadFun(acc))
        t.start()

        time.sleep(0.1)
        self.assertTrue(t.is_alive())  # read is not yet complete

        acc.interrupt()

        t.join(10)
        self.assertFalse(t.is_alive())  # read has completed

    def testBinaryOperators(self):
        for typ in types_to_test:
            # registers don't matter since we do not actually execute any transfer operations
            acc1 = self.dev.getOneDRegisterAccessor(typ, "BOARD/WORD_CLK_MUX")
            acc2 = self.dev.getOneDRegisterAccessor(typ, "BOARD/WORD_CLK_MUX")
            for operator, useForFloat, useForBool, useForStr in binaryOps:

                if typ == str and not useForStr:
                    continue
                if typ == bool and not useForBool:
                    continue
                if (typ == np.float32) or (typ == np.float64) and not useForFloat:
                    continue

                with self.subTest(type=typ, operator=operator):
                    if typ != str:
                        val1 = np.array(value(typ))
                        val2 = np.array(value(typ, val1))
                    else:
                        val1 = value(typ)
                        val2 = value(typ, val1)
                    acc1.set(val1)
                    acc2.set(val2)

                    if not operator.startswith('__i'):
                        expected12 = catchEx(lambda: val1.__getattribute__(operator)(val2))
                        expected21 = catchEx(lambda: val2.__getattribute__(operator)(val1))

                        if typ != str:
                            self.assertTrue(
                                (catchEx(lambda: acc1.__getattribute__(operator)(acc2)) == expected12).all())
                            self.assertTrue(
                                (catchEx(lambda: acc1.__getattribute__(operator)(val2)) == expected12).all())
                            self.assertTrue(
                                (catchEx(lambda: acc2.__getattribute__(operator)(acc1)) == expected21).all())
                            self.assertTrue(
                                (catchEx(lambda: acc2.__getattribute__(operator)(val1)) == expected21).all())
                        else:
                            self.assertTrue(catchEx(lambda: acc1.__getattribute__(operator)(acc2)) == expected12)
                            self.assertTrue(catchEx(lambda: acc1.__getattribute__(operator)(val2)) == expected12)
                            self.assertTrue(catchEx(lambda: acc2.__getattribute__(operator)(acc1)) == expected21)
                            self.assertTrue(catchEx(lambda: acc2.__getattribute__(operator)(val1)) == expected21)

                        acc2.set(val1)
                        expected11 = catchEx(lambda: val1.__getattribute__(operator)(val1))

                        if typ != str:
                            self.assertTrue(
                                (catchEx(lambda: acc1.__getattribute__(operator)(acc2)) == expected11).all())
                            self.assertTrue(
                                (catchEx(lambda: acc1.__getattribute__(operator)(val1)) == expected11).all())
                        else:
                            self.assertTrue(catchEx(lambda: acc1.__getattribute__(operator)(acc2)) == expected11)
                            self.assertTrue(catchEx(lambda: acc1.__getattribute__(operator)(val1)) == expected11)

                    else:
                        # special treatment for assignment operators (+= etc.)
                        non_assigning_operator = '__'+operator[3:]
                        expected12 = catchEx(lambda: val1.__getattribute__(non_assigning_operator)(val2))
                        expected21 = catchEx(lambda: val2.__getattribute__(non_assigning_operator)(val1))

                        self.assertTrue((catchEx(lambda: acc1.__getattribute__(operator)(acc2)) == expected12).all())
                        acc1.set(val1)
                        self.assertTrue((catchEx(lambda: acc1.__getattribute__(operator)(val2)) == expected12).all())
                        acc1.set(val1)
                        self.assertTrue((catchEx(lambda: acc2.__getattribute__(operator)(acc1)) == expected21).all())
                        acc2.set(val2)
                        self.assertTrue((catchEx(lambda: acc2.__getattribute__(operator)(val1)) == expected21).all())

                        acc2.set(val1)
                        expected11 = catchEx(lambda: val1.__getattribute__(non_assigning_operator)(val1))

                        self.assertTrue((catchEx(lambda: acc1.__getattribute__(operator)(acc2)) == expected11).all())
                        acc1.set(val1)
                        self.assertTrue((catchEx(lambda: acc1.__getattribute__(operator)(val1)) == expected11).all())

    def testUnaryOperators(self):
        for typ in types_to_test:
            # registers don't matter since we do not actually execute any transfer operations
            acc = self.dev.getOneDRegisterAccessor(typ, "BOARD/WORD_CLK_MUX")
            for operator, useForFloat, useForBool, useForStr in unaryOps:

                if typ == str and not useForStr:
                    continue
                if typ == bool and not useForBool:
                    continue
                if (typ == np.float32) or (typ == np.float64) and not useForFloat:
                    continue

                with self.subTest(type=typ, operator=operator):
                    for i in range(0, 2):

                        if typ != str:
                            val = np.array(value(typ))
                        else:
                            val = value(typ)
                        acc.set(val)

                        expected = catchEx(lambda: val.__getattribute__(operator)())

                        self.assertEqual(catchEx(lambda: acc.__getattribute__(operator)()), expected)

#####################################################################################################################


if __name__ == '__main__':
    unittest.main()

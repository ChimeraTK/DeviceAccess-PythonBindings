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

types_to_test = [np.int8, np.uint8, np.int16, np.uint16, np.int32,
                 np.uint32, np.int64, np.uint64, np.float32, np.float64, bool, str]

generator_seed = 12345689


def valueAfterConstruct(type):
    """
    Return value-after-construct for the given type.
    """
    if type == str:
        return ""
    if type == bool:
        return False
    return type(0)


def value(type, forceUnequal=None):
    """
    Generate a value of the given type, which differs from the value "forceUnequal". If "forceUnequal" is None,
    the generated value will be different from the value-after-construct.
    """
    global generator_seed

    if forceUnequal is None:
        forceUnequal = valueAfterConstruct(type)

    while True:
        generator_seed += 1
        if type == str:
            value = str(generator_seed)
        elif type == bool:
            value = (generator_seed % 2 == 0)
        else:
            # don't need to cover the full range, so we treat all other's equal
            value = type(generator_seed % 100)

        if value != forceUnequal:
            return type(value)

        generator_seed += 1


class TestScalarRegisterAccessor(unittest.TestCase):

    def setUp(self):
        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        self.dev = da.Device("CARD_WITH_MODULES")
        self.dev.open()
        self.dev.activateAsyncRead()

        # The "backdoor" accessor is used to check the result of write operations resp. provide data for read
        # operations. This kind of creates a "circular reasoning" for the test, which shall be deemed to be acceptable,
        # since the code under test is merely delegating to well tested C++ code of DeviceAccess.
        self.backdoor = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        self.interrupt = self.dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_0")

    def testGetSet(self):
        for type in types_to_test:
            with self.subTest(type=type):
                acc = self.dev.getScalarRegisterAccessor(type, "ADC/WORD_CLK_CNT_1")

                self.assertTrue(acc.get() == [valueAfterConstruct(type)])

                expected = value(type)
                acc.set([expected])

                self.assertTrue(acc.get() == [expected])

                expected = value(type, expected)
                acc.set(expected)

                self.assertTrue(acc.get() == [expected])

    def testRead(self):
        for type in types_to_test:
            with self.subTest(type=type):
                acc = self.dev.getScalarRegisterAccessor(type, "ADC/WORD_CLK_CNT_1")

                expected = [value(type)]
                self.backdoor.set(expected)
                self.backdoor.write()

                acc.read()

                self.assertTrue(acc == expected)

    def testRead_push(self):
        for type in types_to_test:
            with self.subTest(type=type):
                acc = self.dev.getScalarRegisterAccessor(
                    type, "ADC/WORD_CLK_CNT_1_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])
                acc.read()  # initial value

                t = threading.Thread(name='blocking_read', target=lambda acc=acc: acc.read())
                t.start()

                time.sleep(0.1)
                self.assertTrue(t.is_alive())  # read is not yet complete

                expected = [value(type)]
                self.backdoor.set(expected)
                self.backdoor.write()

                self.interrupt.write()
                t.join(1)  # TODO increase to 10s
                self.assertFalse(t.is_alive())  # read has completed

                self.assertTrue(acc == expected, f'{acc} == {expected}')

    def testReadLatest(self):
        for type in types_to_test:
            with self.subTest(type=type):
                acc = self.dev.getScalarRegisterAccessor(
                    type, "ADC/WORD_CLK_CNT_1_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])

                expected = [value(type)]
                self.backdoor.set(expected)
                self.backdoor.write()
                self.interrupt.write()

                retval = acc.readLatest()

                self.assertTrue(acc == expected, f'{acc} == {expected}')
                self.assertTrue(retval)

                retval = acc.readLatest()
                self.assertFalse(retval)

    def testReadNonBlocking(self):
        for type in types_to_test:
            with self.subTest(type=type):
                expected1 = [value(type)]
                self.backdoor.set(expected1)
                self.backdoor.write()

                acc = self.dev.getScalarRegisterAccessor(
                    type, "ADC/WORD_CLK_CNT_1_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])

                expected2 = [value(type, expected1)]
                self.backdoor.set(expected2)
                self.backdoor.write()
                self.interrupt.write()

                retval = acc.readNonBlocking()

                self.assertTrue(acc == expected1, f'{acc} == {expected1}')
                self.assertTrue(retval)

                retval = acc.readNonBlocking()

                self.assertTrue(acc == expected2, f'{acc} == {expected2}')
                self.assertTrue(retval)

                retval = acc.readNonBlocking()
                self.assertFalse(retval)

    def testWrite(self):
        for type in types_to_test:
            with self.subTest(type=type):
                acc = self.dev.getScalarRegisterAccessor(type, "ADC/WORD_CLK_CNT_1")

                expected = [value(type)]
                acc.set(expected)

                acc.write()

                self.backdoor.read()
                self.assertTrue(self.backdoor[0] == int(expected[0]), f'{self.backdoor[0]} == {int(expected[0])}')

    def testWriteDestructively(self):
        for type in types_to_test:
            with self.subTest(type=type):
                acc = self.dev.getScalarRegisterAccessor(type, "ADC/WORD_CLK_CNT_1")

                expected = [value(type)]
                acc.set(expected)

                acc.writeDestructively()

                self.backdoor.read()
                self.assertTrue(self.backdoor[0] == int(expected[0]), f'{self.backdoor[0]} == {int(expected[0])}')

    def testGetName(self):
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        self.assertTrue(acc.getName() == "/ADC/WORD_CLK_CNT_1")

        acc = self.dev.getScalarRegisterAccessor(np.int32, "BOARD.WORD_CLK_MUX_3")
        self.assertTrue(acc.getName() == "/BOARD.WORD_CLK_MUX_3")

    def testGetValueType(self):
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        self.assertTrue(acc.getValueType() == np.int32)

        acc = self.dev.getScalarRegisterAccessor(np.float32, "ADC/WORD_CLK_CNT_1")
        self.assertTrue(acc.getValueType() == np.float32)

    def testGetAccessModeFlags(self):
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        self.assertTrue(acc.getAccessModeFlags() == [])

        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1", accessModeFlags=[da.AccessMode.raw])
        self.assertTrue(acc.getAccessModeFlags() == [da.AccessMode.raw])

    def testGetVersionNumber(self):
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")

        self.assertTrue(acc.getVersionNumber() == da.VersionNumber.getNullVersion())

        myVersion = da.VersionNumber()
        acc.read()

        self.assertTrue(acc.getVersionNumber() != da.VersionNumber.getNullVersion())
        self.assertTrue(acc.getVersionNumber() > myVersion)

    def testIsReadOnly(self):
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        self.assertTrue(acc.isReadOnly() == False)

        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1_RO")
        self.assertTrue(acc.isReadOnly() == True)

    def testIsReadable(self):
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        self.assertTrue(acc.isReadable() == True)

        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1_WO")
        self.assertTrue(acc.isReadable() == False)

    def testIsWriteable(self):
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        self.assertTrue(acc.isWriteable() == True)

        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1_RO")
        self.assertTrue(acc.isWriteable() == False)

    def testIsWriteable(self):
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        self.assertTrue(acc.isInitialised() == True)
        # TODO: unclear how to get an uninitialised accessor!? Maybe remove this function?

    def testDataValidity(self):
        # The backend used for this test cannot deal with data validity, so we just test setDataValidity() and
        # dataValidity() of a single accessor instance
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        self.assertTrue(acc.dataValidity() == da.DataValidity.ok)
        acc.setDataValidity(da.DataValidity.faulty)
        self.assertTrue(acc.dataValidity() == da.DataValidity.faulty)
        acc.setDataValidity(da.DataValidity.ok)
        self.assertTrue(acc.dataValidity() == da.DataValidity.ok)

    def testGetId(self):
        acc1 = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        acc2 = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        acc1_copy = acc1  # testing with a copy barely makes a difference in Python, doesn't hurt anyway
        self.assertTrue(acc1.getId() != acc2.getId())
        self.assertTrue(acc1.getId() == acc1_copy.getId())

    def testInterrupt(self):
        acc = self.dev.getScalarRegisterAccessor(
            np.int32, "ADC/WORD_CLK_CNT_1_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])
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

    def testString(self):
        # test that setting the string accessor to a longer string than currently works
        stringTooShortAcc = self.dev.getScalarRegisterAccessor(str, "ADC/WORD_CLK_CNT_1")
        stringTooShortAcc.set('123')

        self.backdoor.set(123456789)
        self.backdoor.write()

        stringTooShortAcc.read()

        self.assertTrue(stringTooShortAcc == '123456789')


if __name__ == '__main__':
    unittest.main()

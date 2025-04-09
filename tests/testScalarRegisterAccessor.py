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

    def testRead(self):
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        reference = [99]
        self.backdoor.set(reference)
        self.backdoor.write()
        acc.read()
        self.assertTrue(acc == reference)

    def testRead_push(self):
        acc = self.dev.getScalarRegisterAccessor(
            np.int32, "ADC/WORD_CLK_CNT_1_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])
        acc.read()  # initial value

        t = threading.Thread(name='blocking_read', target=lambda acc=acc: acc.read())
        t.start()

        time.sleep(0.1)
        self.assertTrue(t.is_alive())  # read is not yet complete

        self.backdoor.set(17)
        self.backdoor.write()

        self.interrupt.write()
        t.join(1)  # TODO increase to 10s
        self.assertFalse(t.is_alive())  # read has completed

        self.assertTrue(acc[0] == 17)

    def testReadLatest(self):
        acc = self.dev.getScalarRegisterAccessor(
            np.int32, "ADC/WORD_CLK_CNT_1_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])

        reference = [123]
        self.backdoor.set(reference)
        self.backdoor.write()
        self.interrupt.write()

        retval = acc.readLatest()

        self.assertTrue(acc == reference)
        self.assertTrue(retval)

        retval = acc.readLatest()
        self.assertFalse(retval)

    def testReadNonBlocking(self):
        self.backdoor.set(230)
        self.backdoor.write()

        acc = self.dev.getScalarRegisterAccessor(
            np.int32, "ADC/WORD_CLK_CNT_1_INT", accessModeFlags=[da.AccessMode.wait_for_new_data])

        self.backdoor.set(231)
        self.backdoor.write()
        self.interrupt.write()

        retval = acc.readNonBlocking()

        self.assertTrue(acc == 230)
        self.assertTrue(retval)

        retval = acc.readNonBlocking()

        self.assertTrue(acc == 231)
        self.assertTrue(retval)

        retval = acc.readNonBlocking()
        self.assertFalse(retval)

    def testWrite(self):
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        acc.set(120)
        acc.write()
        self.backdoor.read()
        self.assertTrue(self.backdoor == 120)

    def testWriteDestructively(self):
        acc = self.dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
        acc.set(240)
        acc.writeDestructively()
        self.backdoor.read()
        self.assertTrue(self.backdoor == 240)

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

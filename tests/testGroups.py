#!/usr/bin/env python3
# SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
# SPDX-License-Identifier: LGPL-3.0-or-later

import random
import sys
import threading
import unittest
import numpy as np
import os
import time
import multiprocessing


# fmt: off
# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module. Formatting is switched off,
# so the import is not sorted into the others.
sys.path.insert(0, os.path.abspath(os.path.join(os.curdir,"..")))
import deviceaccess as da
# fmt: on


class TestGroups(unittest.TestCase):

    def checkUntilTimeout(self, func: callable, timeout: float = 10.0, interval: float = 0.01) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if func():
                return True
            time.sleep(interval)
        return False

    def setUp(self):
        da.setDMapFilePath("deviceInformation/testCrate.dmap")
        self.dev: da.Device = da.Device("TEST_CARD")
        self.dev.open()
        self.dev.activateAsyncRead()

        self.pushScalar0: da.ScalarRegisterAccessor = self.dev.getScalarRegisterAccessor(np.int32, "INT32_TEST.SCALAR")
        self.pushScalar0_copy: da.ScalarRegisterAccessor = self.dev.getScalarRegisterAccessor(
            np.int32, "INT32_TEST.SCALAR")
        self.pushScalar1: da.ScalarRegisterAccessor = self.dev.getScalarRegisterAccessor(
            np.int32, "FLOAT32_TEST.SCALAR")

        self.scalar0: da.ScalarRegisterAccessor = self.dev.getScalarRegisterAccessor(
            np.int32, "GROUP_TEST.SCALAR0", 0, [da.AccessMode.wait_for_new_data])
        self.scalar1: da.ScalarRegisterAccessor = self.dev.getScalarRegisterAccessor(
            np.int32, "GROUP_TEST.SCALAR1", 0, [da.AccessMode.wait_for_new_data])
        self.scalar2: da.ScalarRegisterAccessor = self.dev.getScalarRegisterAccessor(
            np.int32, "GROUP_TEST.SCALAR2", 0, [da.AccessMode.wait_for_new_data])

        self.scalar0_writeable: da.ScalarRegisterAccessor = self.dev.getScalarRegisterAccessor(
            np.int32, "GROUP_TEST.SCALAR0/DUMMY_WRITEABLE")
        self.scalar1_writeable: da.ScalarRegisterAccessor = self.dev.getScalarRegisterAccessor(
            np.int32, "GROUP_TEST.SCALAR1/DUMMY_WRITEABLE")
        self.scalar2_writeable: da.ScalarRegisterAccessor = self.dev.getScalarRegisterAccessor(
            np.int32, "GROUP_TEST.SCALAR2/DUMMY_WRITEABLE")

        self.interrupt0: da.VoidRegisterAccessor = self.dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_0")
        self.interrupt1: da.VoidRegisterAccessor = self.dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_1")
        self.interrupt2: da.VoidRegisterAccessor = self.dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_2")

    def tearDown(self) -> None:
        self.dev.close()

    def testReadAnyGroupsPushType(self):
        registers = [self.scalar0, self.scalar1, self.scalar2]

        group: da.ReadAnyGroup = da.ReadAnyGroup()
        for acc in registers:
            acc.readLatest()

        group.add(self.scalar0)
        group.add(self.scalar1)
        group.finalise()  # don't add third one, so we can test that it is not read

        self.assertRaises(RuntimeError, lambda: group.add(self.scalar2))  # can't add after finalise

        # no data should be available, so the id should be invalid
        id = group.readAnyNonBlocking()
        self.assertFalse(id.isValid())

        # reading should raise, as the accessor is now part of a group
        self.assertRaises(RuntimeError, lambda: self.scalar0.read())

        # test readAny
        value = random.randrange(0, 100)
        self.scalar0_writeable.setAndWrite(value)
        self.interrupt0.write()
        id: da.TransferElementID = group.readAny()
        self.assertTrue(id.isValid())
        self.assertEqual(id, self.scalar0.getId())
        self.assertEqual(self.scalar0.get(), value)

        # test readAnyNonBlocking after data is available
        value = random.randrange(0, 100)
        self.scalar1_writeable.setAndWrite(value)
        self.interrupt1.write()
        self.assertTrue(self.checkUntilTimeout(lambda: group.readAnyNonBlocking().isValid()))

        # test readuntil, filling the queue
        for i in range(3):
            self.scalar1_writeable.setAndWrite(i)
            self.interrupt1.write()
            time.sleep(0.01)

        self.scalar0_writeable.setAndWrite(3)
        self.interrupt0.write()
        time.sleep(0.01)
        self.scalar1_writeable.setAndWrite(49)
        self.interrupt1.write()
        time.sleep(0.01)
        group.readUntil(self.scalar0)
        self.assertEqual(self.scalar0.get(), 3)

        # check readUntilAll with a list of acccessors
        self.scalar1_writeable.setAndWrite(51)
        self.interrupt1.write()
        time.sleep(0.01)
        self.scalar0_writeable.setAndWrite(59)
        self.interrupt0.write()
        time.sleep(0.01)
        group.readUntilAll([self.scalar0, self.scalar1])
        self.assertEqual(self.scalar0.get(), 59)
        self.assertEqual(self.scalar1.get(), 51)

        # check readUntilAll with a list of ids
        self.scalar1_writeable.setAndWrite(510)
        self.interrupt1.write()
        time.sleep(0.01)
        self.scalar0_writeable.setAndWrite(590)
        self.interrupt0.write()
        time.sleep(0.01)
        group.readUntilAll([self.scalar0.getId(), self.scalar1.getId()])
        self.assertEqual(self.scalar0.get(), 590)
        self.assertEqual(self.scalar1.get(), 510)

        # check waitAny and notifications
        self.scalar1_writeable.setAndWrite(23)
        self.interrupt1.write()
        time.sleep(0.01)
        notification: da.ReadAnyGroupNotification = group.waitAny()
        self.assertEqual(notification.getId(), self.scalar1.getId())
        self.assertEqual(self.scalar1.get(), 510)
        self.assertTrue(notification.isReady())
        notification.accept()
        self.assertFalse(notification.isReady())
        self.assertEqual(self.scalar1.get(), 23)
        notification: da.ReadAnyGroupNotification = group.waitAnyNonBlocking()
        self.assertEqual(self.scalar1.get(), 23)
        self.assertRaises(RuntimeError, lambda: notification.accept())  # can't accept twice

    def testReadAnyGroupsPollAndPushType(self):
        pollgroup: da.ReadAnyGroup = da.ReadAnyGroup()
        self.pushScalar0.readLatest()
        self.pushScalar1.readLatest()
        self.scalar0.readLatest()
        self.pushScalar0_copy.readLatest()  # not in the group
        pollgroup.add(self.pushScalar0)
        pollgroup.add(self.pushScalar1)
        pollgroup.add(self.scalar0)
        pollgroup.finalise()

        self.assertEqual(self.pushScalar0.get(), 0)
        self.assertEqual(self.pushScalar0_copy.get(), 0)
        self.pushScalar0_copy.setAndWrite(42)
        self.assertEqual(self.pushScalar0.get(), 0)
        self.assertEqual(self.pushScalar0_copy.get(), 42)
        self.interrupt0.write()
        # check mixed group
        pollgroup.readAny()
        self.assertEqual(self.pushScalar0.get(), 42)
        # only update polltypes
        self.scalar0_writeable.setAndWrite(24)
        self.interrupt0.write()
        self.pushScalar0_copy.setAndWrite(412)
        pollgroup.processPolled()
        self.assertEqual(self.pushScalar0.get(), 412)
        self.assertEqual(self.scalar0.get(), 0)  # should not be updated, as it is not push type


if __name__ == '__main__':
    unittest.main()

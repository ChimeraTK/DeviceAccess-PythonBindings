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


class TestPushType(unittest.TestCase):

    def testCorrectInterupt(self):

        da.setDMapFilePath("deviceInformation/push.dmap")
        dev = da.Device("SHARED_RAW_DEVICE")
        dev.open()
        dev.activateAsyncRead()

        readAcc = dev.getOneDRegisterAccessor(
            np.int32, "MODULE1/TEST_AREA_PUSH", 0, 0, [da.AccessMode.wait_for_new_data])
        interruptAcc = dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_2_3")
        readAcc.read()  # first read should be non-blocking
        self.assertFalse(readAcc.readNonBlocking())
        interruptAcc.write()
        readAcc.read()  # should work, because interupt has been triggered
        #  if interrupt does not triggger: test will fail, because timeout will hit
        # interupt is set at CMakeList.txt

    def testThreadedRead(self):

        def blockingRead(readAcc, barrier):
            barrier.wait()
            readAcc.read()
            barrier.wait()

        da.setDMapFilePath("deviceInformation/push.dmap")
        dev = da.Device("SHARED_RAW_DEVICE")
        dev.open()
        dev.activateAsyncRead()

        readAcc = dev.getOneDRegisterAccessor(
            np.int32, "MODULE1/TEST_AREA_PUSH", 0, 0, [da.AccessMode.wait_for_new_data])
        interruptAcc = dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_2_3")
        readAcc.read()  # first read is always non-blocking
        barrier = threading.Barrier(2)

        readThread = threading.Thread(
            target=blockingRead, args=(readAcc, barrier), daemon=True)
        readThread.start()

        barrier.wait()
        interruptAcc.write()
        barrier.reset()
        # if blockingRead is not at the second barrier after 2 sec, the barrier
        # will throw, so the test will fail.
        barrier.wait(timeout=2)

    def testCorrectWrite(self):
        da.setDMapFilePath("deviceInformation/push.dmap")
        dev = da.Device("SHARED_RAW_DEVICE")
        dev.open()
        dev.activateAsyncRead()

        writeAcc = dev.getOneDRegisterAccessor(np.int32, "MODULE1/TEST_AREA")
        arr1to10 = np.array([i for i in range(1, 11)], dtype=np.int32)
        writeAcc += arr1to10
        writeAcc.write()

        readAcc = dev.getOneDRegisterAccessor(
            np.int32, "MODULE1/TEST_AREA_PUSH", 0, 0, [da.AccessMode.wait_for_new_data])
        readAcc.read()
        # check if readAcc has  the correct values:
        self.assertListEqual(list(arr1to10), list(readAcc.view())) #

        # double values of writeAccReg
        writeAcc += arr1to10
        writeAcc.write()
        # no new results, as interrupt has not been triggered:
        self.assertFalse(readAcc.readNonBlocking())
        interruptAcc = dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_2_3")
        interruptAcc.write()

        readAcc.read()
        # Check if values are also doubled in readAcc
        self.assertListEqual(list(arr1to10*2), list(readAcc.view()))



if __name__ == '__main__':
    unittest.main()

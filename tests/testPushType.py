#!/usr/bin/env ${python_interpreter}
import mtca4udeviceaccess
import deviceaccess as da
import sys
import unittest
import numpy as np
import os

# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module.
sys.path.insert(0, os.path.abspath(os.curdir))


class TestAsyncWrite(unittest.TestCase):

    def testExceptionWhenOpening(self):
        da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
        dev = da.Device("CARD_WITH_MODULES")
        dev.open()
        writeAcc = dev.getOneDRegisterAccessor(np.int32, "MODULE1/TEST_AREA")
        readAcc = dev.getOneDRegisterAccessor(
            np.int32, "MODULE1/TEST_AREA_PUSH")
        interruptAcc = dev.getVoidRegisterAccessor(
            "MODULE1/DATA_READY", [da.AccessMode.wait_for_new_data])


if __name__ == '__main__':
    unittest.main()

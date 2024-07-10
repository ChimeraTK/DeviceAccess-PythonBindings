#!/usr/bin/env python3
# SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
# SPDX-License-Identifier: LGPL-3.0-or-later

from concurrent.futures import thread
import sys
import unittest
import numpy as np
import os
import threading
from time import sleep


# fmt: off
# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module. Formatting is switched off,
# so the import is not sorted into the others.
sys.path.insert(0, os.path.abspath(os.path.join(os.curdir,"..")))
import deviceaccess as da
# fmt: on


class TestRegisterCatalogue(unittest.TestCase):

    def setUp(self):
        da.setDMapFilePath("deviceInformation/push.dmap")
        self.dev = da.Device("SHARED_RAW_DEVICE")

    def tearDown(self) -> None:
        pass

    def testIterator(self):
        # Test to see if the iterator function works in a for loop
        rc = self.dev.getRegisterCatalogue()
        registers_in_num_dev = ['/BOARD/WORD_FIRMWARE',
                                '/BOARD/WORD_COMPILATION',
                                '/APP0/WORD_STATUS',
                                '/APP0/WORD_SCRATCH',
                                '/APP0/MODULE0',
                                '/APP0/MODULE1',
                                '/MODULE0/WORD_USER1',
                                '/MODULE0/WORD_USER2',
                                '/MODULE1/WORD_USER1',
                                '/MODULE1/WORD_USER2',
                                '/MODULE1/TEST_AREA',
                                '/MODULE1/TEST_AREA_PUSH',
                                '/MODULE1/DATA_READY']
        element_counter = 0
        for i, iterated_register in enumerate(rc):
            # should not have more elements
            self.assertTrue(i < len(registers_in_num_dev))
            # should list same names
            self.assertEqual(iterated_register.getRegisterName(), registers_in_num_dev[i])
            element_counter += 1

        # should not have fewer elements
        self.assertEqual(element_counter, len(registers_in_num_dev))


if __name__ == '__main__':
    unittest.main()

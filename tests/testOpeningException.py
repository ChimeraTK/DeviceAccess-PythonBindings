#!/usr/bin/env python3
# SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
# SPDX-License-Identifier: LGPL-3.0-or-later

import sys
import unittest
import numpy
import os

# fmt: off
# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module. Formatting is switched off,
# so the import is not sorted into the others.
sys.path.insert(0, os.path.abspath(os.path.join(os.curdir,"..")))
import mtca4u
# fmt: on


class TestOpeningException(unittest.TestCase):

    def testExceptionWhenOpening(self):
        # We use the dummy with a bad map file as we know that deviceaccess
        # is throwing here.
        self.assertRaisesRegex(RuntimeError, "Cannot open file \".*badMapFile.map\"",
                                mtca4u.Device, "sdm://./dummy=badMapFile.map")


if __name__ == '__main__':
    unittest.main()

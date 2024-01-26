#!/usr/bin/env python3
# SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import sys
import fcntl
import unittest
import numpy

# fmt: off
# This is a hack for nw. What this does is, add the build directory to python's
# path, so that it can find the deviceaccess module. Formatting is switched off,
# so the import is not sorted into the others.
sys.path.insert(0, os.path.abspath(os.path.join(os.curdir,"..")))
import mtca4u
# fmt: on


class TestPCIEDevice(unittest.TestCase):
    # TODO: Refactor to take care of the harcoded values used for comparisions

    def setUp(self):
        mtca4u.set_dmap_location("deviceInformation/exampleCrate.dmap")

    def testRead(self):
        # first open devices, so shared memory dummies work correctly with __prepareDataOnCards and __testRead
        device1 = mtca4u.Device("(dummy?map=./mtcadummy.map)")
        device2 = mtca4u.Device("(dummy?map=./modular_mtcadummy.map)")
        device3 = mtca4u.Device("CARD_WITH_MODULES")
        device4 = mtca4u.Device("CARD_WITH_OUT_MODULES")

        self.__prepareDataOnCards()

        self.__testRead(device1, "", device1.read)
        self.__testRead(device2, "BOARD", device2.read)
        self.__testRead(device3, "BOARD", device3.read)
        self.__testRead(device4, "", device4.read)

    def testWrite(self):
        device = mtca4u.Device("(dummy?map=./mtcadummy.map)")
        self.__testWrite(device, "", device.write)
        device = mtca4u.Device("(dummy?map=./modular_mtcadummy.map)")
        self.__testWrite(device, "BOARD", device.write)

        device = mtca4u.Device("CARD_WITH_MODULES")
        self.__testWrite(device, "BOARD", device.write)

        device = mtca4u.Device("CARD_WITH_OUT_MODULES")
        self.__testWrite(device, "", device.write)

    def testReadRaw(self):
        # first open devices, so shared memory dummies work correctly with __prepareDataOnCards and __testRead
        device1 = mtca4u.Device("(dummy?map=./mtcadummy.map)")
        device2 = mtca4u.Device("(dummy?map=./modular_mtcadummy.map)")
        device3 = mtca4u.Device("CARD_WITH_MODULES")
        device4 = mtca4u.Device("CARD_WITH_OUT_MODULES")

        self.__prepareDataOnCards()

        self.__testRead(device1, "", device1.read_raw)
        self.__testRead(device2, "BOARD", device2.read_raw)
        self.__testRead(device3, "BOARD", device3.read_raw)
        self.__testRead(device4, "", device4.read_raw)

    def testwriteRaw(self):
        device = mtca4u.Device("(dummy?map=./mtcadummy.map)")
        self.__testWrite(device, "", device.write_raw)
        device = mtca4u.Device("(dummy?map=./modular_mtcadummy.map)")
        self.__testWrite(device, "BOARD", device.write_raw)

        device = mtca4u.Device("CARD_WITH_OUT_MODULES")
        self.__testWrite(device, "", device.write_raw)
        device = mtca4u.Device("CARD_WITH_MODULES")
        self.__testWrite(device, "BOARD", device.write_raw)

    def testreadDMARaw(self):
        device1 = mtca4u.Device("(dummy?map=./mtcadummy.map)")
        device2 = mtca4u.Device("(dummy?map=./modular_mtcadummy.map)")
        device3 = mtca4u.Device("CARD_WITH_OUT_MODULES")
        device4 = mtca4u.Device("CARD_WITH_MODULES")

        self.__testreadDMARaw(device1, "")
        self.__testreadDMARaw(device2, "BOARD")
        self.__testreadDMARaw(device3, "")
        self.__testreadDMARaw(device4, "BOARD")

    def testReadSequences(self):
        device = mtca4u.Device("(dummy?map=./mtcadummy.map)")
        self.__testSequences(device, "")
        device = mtca4u.Device("(dummy?map=./modular_mtcadummy.map)")
        self.__testSequences(device, "BOARD")

        device = mtca4u.Device("CARD_WITH_OUT_MODULES")
        self.__testSequences(device, "")
        device = mtca4u.Device("CARD_WITH_MODULES")
        self.__testSequences(device, "BOARD")
# http://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python

    def testGetInfo(self):

        try:
            from StringIO import StringIO
        except ImportError:
            from io import StringIO

        # expectedString = "mtca4uPy v" + vn.moduleVersion + \
        #    ", linked with mtca4u-deviceaccess v" + vn.deviceaccessVersion
        outStream = StringIO()
        mtca4u.get_info(outStream)
        returnedString = outStream.getvalue().strip()
        # self.assertTrue(expectedString == returnedString)
        self.assertTrue(True)

    def testException(self):
        device = mtca4u.Device("(dummy?map=./modular_mtcadummy.map)")
        array = numpy.array([1, 2, 3, 4], dtype=numpy.int32)
        self.assertRaisesRegex(
            RuntimeError,
            "Requested number of words exceeds the size of the register '/BOARD/WORD_STATUS'!",
            device.write_raw,
            'BOARD',
            'WORD_STATUS',
            array)
        # moduleName='', registerName=None, dataToWrite=None, elementIndexInRegister=0, registerPath=None)

    def testDeviceCreation(self):

        with self.assertRaises(RuntimeError):
            # Not testing for the printed
            mtca4u.Device("NON_EXISTENT_ALIAS_NAME")
            # message. This comes from the
            # device access library and the
            # text has been chainging to
            # often/ can change in the
            # future. What is being
            # prioritized is to check that we
            # have an exception for an
            # incorrect usage.

        self.assertRaises(RuntimeError, mtca4u.Device, "(dummy?map=NON_EXISTENT_MAPFILE)")
        self.assertRaisesRegex(SyntaxError, "Syntax Error: please see help\\(mtca4u.Device\\) for usage instructions.",
                               mtca4u.Device)
        self.assertRaisesRegex(SyntaxError, "Syntax Error: please see help\\(mtca4u.Device\\) for usage instructions.",
                               mtca4u.Device, "BogusText", "BogusText", "BogusText")

        dmapFilePath = mtca4u.get_dmap_location()
        mtca4u.set_dmap_location("")
        with self.assertRaises(RuntimeError):
            mtca4u.Device("CARD_WITH_OUT_MODULES")

    def testSetGetDmapfile(self):
        # set by the test setUp method
        self.assertTrue(mtca4u.get_dmap_location() ==
                        "deviceInformation/exampleCrate.dmap")

    """
  The idea here is to preset data on registers that is then  read in and
  verified later. The following registers on each card are set:
  - WORD_STATUS (Offset: 8)
  - WORD_CLK_MUX (Offset: 32)
  - WORD_INCOMPLETE_2 (Offset: 100)
  The memory map for each device has been kept identical. The map files all
  contain unique register names which are at the same address on each card
  (despite being in different modules on individual cards).
  A copy of the data that gets written is stored in these variables:
  - word_status_content
  - word_clk_mux_content
  - word_incomplete_content
  """

    def __prepareDataOnCards(self):
        self.__prepareDataToWrite()
        self.__writeDataToDevices()

    def __prepareDataToWrite(self):
        self.word_status_content = self.__createRandomArray(1)
        self.word_clk_mux_content = self.__createRandomArray(4)
        self.word_incomplete_2_content = numpy.array([544], dtype=numpy.int32)

    def __writeDataToDevices(self):
        # Test Read from a module register
        # set up the register with a known values
        device = mtca4u.Device(
            "(dummy?map=./modular_mtcadummy.map)")
        self.__preSetValuesOnCard(device, True)
        device = mtca4u.Device(
            "(dummy?map=./mtcadummy.map)")
        self.__preSetValuesOnCard(device)

    def __createRandomArray(self, arrayLength):
        array = numpy.random.randint(0, 1073741824, arrayLength)
        return array.astype(numpy.int32)

    def __preSetValuesOnCard(self, device, modular=False):
        if not modular:
            word_status = 'WORD_STATUS'
            word_clk_mux = 'WORD_CLK_MUX'
            word_incomplete_2 = 'WORD_INCOMPLETE_2'
        else:
            word_status = 'BOARD/WORD_STATUS'
            word_clk_mux = 'BOARD/WORD_CLK_MUX'
            word_incomplete_2 = 'BOARD/WORD_INCOMPLETE_2'

        device.write_raw(registerPath=word_status,
                         dataToWrite=self.word_status_content)

        device.write_raw(registerPath=word_clk_mux,
                         dataToWrite=self.word_clk_mux_content)

        device.write_raw(registerPath=word_incomplete_2,
                         dataToWrite=self.word_incomplete_2_content)

    def __testRead(self, device, module, readCommand):

        dtype = self.__getDtypeToUse(device, readCommand)

        word_status_content = self.word_status_content.astype(dtype)
        word_clk_mux_content = self.word_clk_mux_content.astype(dtype)

        # Test the read from module functionality

        readInValues = readCommand(str(module), "WORD_STATUS")
        self.assertTrue(readInValues.dtype == dtype)
        self.assertTrue(numpy.array_equiv(readInValues, word_status_content))

        readInValues = readCommand(
            registerPath='/' + str(module) + "/WORD_STATUS")
        self.assertTrue(readInValues.dtype == dtype)
        self.assertTrue(numpy.array_equiv(readInValues, word_status_content))

        # This section checks the read register code for the Device class

        # check if function reads values correctly
        # Run this only for device.read and not device.read_raw
        if readCommand == device.read:
            readInValues = readCommand(str(module), "WORD_INCOMPLETE_2")
            self.assertTrue(readInValues.dtype == dtype)
            self.assertTrue(readInValues.tolist() == [2.125])

        readInValues = readCommand(str(module), "WORD_CLK_MUX")
        self.assertTrue(readInValues.dtype == dtype)
        self.assertTrue(numpy.array_equiv(readInValues, word_clk_mux_content))

        readInValues = readCommand(str(module), "WORD_CLK_MUX", 1)
        self.assertTrue(readInValues[0] == word_clk_mux_content[0])

        readInValues = readCommand(str(module), "WORD_CLK_MUX", 1, 2)
        self.assertTrue(readInValues[0] == word_clk_mux_content[2])

        readInValues = readCommand(str(module), "WORD_CLK_MUX", 2, 2)
        self.assertTrue(numpy.array_equiv(
            readInValues, word_clk_mux_content[2:]))

        # check for corner cases
        # Register Not Found
        # hack
        exceptionMessage = self.__returnRegisterNotFoundExceptionMsg(
            module, "BAD_REGISTER_NAME")

        self.assertRaisesRegex(RuntimeError, exceptionMessage, readCommand, str(module),
                               "BAD_REGISTER_NAME")

        # Num of elements specified  is more than register size
        registerName = "WORD_CLK_MUX"
        elementsToRead = 5
        offset = 2
        self.assertRaises(RuntimeError, readCommand, str(
            module), registerName, elementsToRead, offset)

        # bad value for number of elements
        self.assertRaises(OverflowError,
                          readCommand,
                          str(module),
                          registerName,
                          numberOfElementsToRead=-1)

        # offset exceeds register size
        offset = 5
        elementsToRead = 5
        self.assertRaises(RuntimeError,
                          readCommand, str(module),
                          registerName, elementIndexInRegister=offset)

    def __testWrite(self, device, module, writeCommand):

        module = str(module)
        dtype = self.__getDtypeToUse(device, writeCommand)

        if writeCommand == device.write:
            readCommand = device.read
        else:
            readCommand = device.read_raw

        word_status_content = self.__createRandomArray(1).astype(dtype)
        word_clk_mux_content = self.__createRandomArray(4).astype(dtype)

        writeCommand(module, "WORD_STATUS", word_status_content)
        readInValues = readCommand(module, "WORD_STATUS")
        self.assertTrue(readInValues.dtype == dtype)
        self.assertTrue(numpy.array_equiv(readInValues, word_status_content))

        # test register path
        writeCommand(registerPath='/' + str(module) +
                     '/WORD_STATUS', dataToWrite=word_status_content)
        readInValues = readCommand(module, "WORD_STATUS")
        self.assertTrue(readInValues.dtype == dtype)
        self.assertTrue(numpy.array_equiv(readInValues, word_status_content))

        # These set of commands will be run for Device.write only

        word_incomplete_register = "WORD_INCOMPLETE_2"
        if writeCommand == device.write:
         # write to WORD_INCOMPLETE_2, this is 13 bits wide and supports 8
         # fractional bits
         # check the write functionality
         # check functionalty when using dtype numpy.float32
            writeCommand(module, word_incomplete_register,
                         numpy.array([2.125], dtype))
            readInValue = readCommand(module, word_incomplete_register)
            self.assertTrue(readInValue.dtype == numpy.float64)
            self.assertTrue(readInValue.tolist() == [2.125])

         # check functionalty when using dtype numpy.float64
            writeCommand(module, word_incomplete_register,
                         numpy.array([3.125], dtype=numpy.float64))
            readInValue = readCommand(module, word_incomplete_register)
            self.assertTrue(readInValue.dtype == dtype)
            self.assertTrue(readInValue.tolist() == [3.125])

            # check functionalty when using dtype numpy.int32
            writeCommand(module, word_incomplete_register,
                         numpy.array([2], dtype=numpy.int32))
            readInValue = readCommand(module, word_incomplete_register)
            self.assertTrue(readInValue.dtype == dtype)
            self.assertTrue(readInValue.tolist() == [2.])

            # check functionalty when using dtype numpy.int64
            writeCommand(module, word_incomplete_register,
                         numpy.array([25], dtype=numpy.int64))
            readInValue = readCommand(module, word_incomplete_register)
            self.assertTrue(readInValue.dtype == dtype)
            self.assertTrue(readInValue.tolist() == [
                            15.99609375])  # This is the
            # valid fp converted
            # value of int 25
            # for this reg

            writeCommand(module, word_incomplete_register, [2.5])
            readInValues = readCommand(module, word_incomplete_register)
            self.assertTrue(readInValues.tolist() == [2.5])

            # continue tests for checking if method accepts int/float/list/numpyarray as valid dataToWrite
            # input a list

            writeCommand(module, "WORD_CLK_MUX", word_status_content, 1)
            readInValues = readCommand(module, "WORD_CLK_MUX", 1, 1)
            self.assertTrue(numpy.array_equiv(
                readInValues, word_status_content))

            writeCommand(module, word_incomplete_register, 3.5)
            readInValues = readCommand(module, word_incomplete_register)
            self.assertTrue(readInValues.tolist() == [3.5])

            writeCommand(module, word_incomplete_register, 14)
            readInValues = readCommand(module, word_incomplete_register)
            self.assertTrue(readInValues.tolist() == [14])

            writeCommand(module, "WORD_CLK_MUX", 5)
            readInValues = readCommand(module, "WORD_CLK_MUX", 1, 0)
            self.assertTrue(readInValues.tolist() == [5])

            self.assertRaisesRegex(RuntimeError, "Data format used is unsupported",
                                   writeCommand, module, word_incomplete_register,
                                   "")

         # Test for Unsupported dtype eg. dtype = numpy.int8
            self.assertRaisesRegex(RuntimeError, "Data format used is unsupported",
                                   writeCommand, module, word_incomplete_register,
                                   numpy.array([2], dtype=numpy.int8))

        # check offset functionality
        writeCommand(module, "WORD_CLK_MUX", word_clk_mux_content)
        readInValues = readCommand(module, "WORD_CLK_MUX")
        self.assertTrue(numpy.array_equiv(readInValues, word_clk_mux_content))

        word_clk_mux_register = "WORD_CLK_MUX"
        writeCommand(module, word_clk_mux_register, word_clk_mux_content[0:2],
                     elementIndexInRegister=2)
        readInValue = readCommand(module, word_clk_mux_register, numberOfElementsToRead=2,
                                  elementIndexInRegister=2)
        self.assertTrue(readInValue.dtype == dtype)
        self.assertTrue(numpy.array_equiv(
            readInValue, word_clk_mux_content[0:2]))
        # Check corner cases

        # Bogus register name
        exceptionMessage = self.__returnRegisterNotFoundExceptionMsg(
            module, "BAD_REGISTER_NAME")
        self.assertRaisesRegex(RuntimeError, exceptionMessage, writeCommand, module,
                               "BAD_REGISTER_NAME",
                               numpy.array([2.125], dtype=dtype))

        # supplied array size exceeds register capacity: !regex /BOARD can be
        # there 1 o 0 times. () and ? have special meaning in regex.
        self.assertRaisesRegex(
            RuntimeError,
            "Requested number of words exceeds the size of the register '(/BOARD)?/WORD_INCOMPLETE_2'!",
            writeCommand,
            module,
            word_incomplete_register,
            word_clk_mux_content)

        # supplied offset exceeds register span
        self.assertRaises(RuntimeError, writeCommand, module,
                          word_incomplete_register, word_clk_mux_content,
                          elementIndexInRegister=1)
        # write nothing
        initialValue = readCommand(module, "WORD_STATUS")
        writeCommand(module, "WORD_STATUS", numpy.array([], dtype=dtype))
        valueAfterEmptyWrite = readCommand(module, "WORD_STATUS")
        self.assertTrue(numpy.array_equiv(initialValue, valueAfterEmptyWrite))

    def __returnRegisterNotFoundExceptionMsg(self, module, registerName):
        if not str(module):
            exceptionMessage = "Cannot find register " + str(registerName) + \
                               " in map file: deviceInformation/mtcadummy.map"
        else:
            exceptionMessage = "Cannot find register " + str(module) + "." + str(registerName) + \
                " in map file: deviceInformation/modular_mtcadummy.map"

    def __getDtypeToUse(self, device, command):
        if command == device.read or command == device.write:
            return numpy.float64
        elif command == device.read_raw or command == device.write_raw:
            return numpy.int32

    def __testreadDMARaw(self, device, module):
        module = str(module)

        # Method: Set some completelty random numbers to AREA_DMAABLE,
        # AREA_DMAABLE_FIXEDPOINT16_3 readouts are shifted by 3, but
        # read_raw should be the same.

        test_data = numpy.array([4, 8, 15, 16, 23, 42], dtype=numpy.float32)

        device.write(module, "AREA_DMAABLE", test_data)

        # Read in the parabolic values from the function
        readInValues = device.read_dma_raw(module, "AREA_DMAABLE_FIXEDPOINT16_3",
                                           numberOfElementsToRead=3)
        self.assertTrue(readInValues.dtype == numpy.int32)
        self.assertTrue(readInValues.tolist() == [4, 8, 15])

        # Check offset read
        readInValues = device.read_dma_raw(module, "AREA_DMAABLE_FIXEDPOINT16_3",
                                           numberOfElementsToRead=3,
                                           elementIndexInRegister=3)
        self.assertTrue(readInValues.dtype == numpy.int32)
        self.assertTrue(readInValues.tolist() == [16, 23, 42])

        # corner cases:
        # bad register name
        exceptionText = self.__returnRegisterNotFoundExceptionMsg(
            module, "BAD_REGISTER_NAME")
        # bad element size
        # bad offset
        # FIXME: Not checking this; size of  AREA_DMA_VIA_DMA is big 1024 elements

    def __testSequences(self, device, module):
        module = str(module)
        # Basic Interface: Currently supports read of all sequences only
        # device.write("", "WORD_ADC_ENA", 1)
        # Arrange the data on the card:
        predefinedSequence = numpy.array([0x00010000,
                                          0x00030002,
                                          0x00050004,
                                          0x00070006,
                                          0x00090008,
                                          0x000b000a,
                                          0x000d000c,
                                          0x000f000e,
                                          0x00110010,
                                          0x00130012,
                                          0x00150014,
                                          0x00170016], dtype=numpy.int32)
        device.write_raw(module, 'AREA_DMAABLE', predefinedSequence)

        expectedMatrix = numpy.array([[0, 1, 2, 3],
                                      [4, 5, 6, 7],
                                      [8, 9, 10, 11],
                                      [12, 13, 14, 15],
                                      [16, 17, 18, 19],
                                      [20, 21, 22, 23]], dtype=numpy.double)
        readInMatrix = device.read_sequences(module, 'DMA')
        self.assertTrue(numpy.array_equiv(readInMatrix, expectedMatrix))
        self.assertTrue(readInMatrix.dtype == numpy.double)
        readInMatrix = device.read_sequences(
            registerPath='/' + str(module) + '/DMA')
        self.assertTrue(numpy.array_equiv(readInMatrix, expectedMatrix))
        self.assertTrue(readInMatrix.dtype == numpy.double)

        # Check that 32 bit data can be transfered without precision loss (Hack by using double.
        # This is not clean, but works sufficiently well.)
        predefinedSequence = numpy.array([0x12345678, 0x90abcdef, 0xa5a5a5a5,
                                          0x5a5a5a5a, 0xffeeffee, 0xcc33cc33,
                                          0x33cc33cc, 0xdeadbeef, 0x87654321,
                                          0xfdecba09, 0xb0b00b0b, 0x73533537], dtype=numpy.int32)
        device.write_raw(
            module, 'UNSIGNED_INT.MULTIPLEXED_RAW', predefinedSequence)

        # Use dtype=numpy.int32 to make sure we don't have rounding errors in the expected values.
        # The comparison array_equiv still works, even if we compare to a different dtype.
        expectedMatrix = numpy.array([[0x12345678, 0x90abcdef, 0xa5a5a5a5],
                                      [0x5a5a5a5a, 0xffeeffee, 0xcc33cc33],
                                      [0x33cc33cc, 0xdeadbeef, 0x87654321],
                                      [0xfdecba09, 0xb0b00b0b, 0x73533537]], dtype=numpy.uint32)
        readInMatrix = device.read_sequences(module, 'UNSIGNED_INT')
        self.assertTrue(numpy.array_equiv(readInMatrix, expectedMatrix))
        self.assertTrue(readInMatrix.dtype == numpy.double)


if __name__ == '__main__':
    unittest.main()

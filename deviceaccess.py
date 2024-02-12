# SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
This module offers the functionality of the DeviceAccess C++ library for python.

The ChimeraTK DeviceAccess library provides an abstract interface for register
based devices. Registers are identified by a name and usually accessed though
an accessor object. Since this library also allows access to other control
system applications, it can be understood as the client library of the
ChimeraTK framework.

More information on ChimeraTK can be found at the project's
`github.io <https://chimeratk.github.io/>`_.
"""

from __future__ import annotations

from typing import Sequence, Union
import _da_python_bindings as pb
import numpy as np
from _da_python_bindings import AccessMode, DataValidity, TransferElementID, VersionNumber
import abc
import functools

#######################################################################################################################


def setDMapFilePath(dmapFilePath: str) -> None:
    """
    Set the location of the dmap file.

    The library will parse this dmap file for the device(alias) lookup.
    Relative or absolute path of the dmap file (directory and file name).

    Examples
    --------
    Setting the location of the dmap file
      >>> import deviceaccess as da
      >>> da.setDMapFilePath('deviceInformation/exampleCrate.dmap')
      >>> dmap_path = da.getDMapFilePath()
      >>> print(dmap_path)
      deviceInformation/exampleCrate.dmap
    """
    pb.setDmapFile(dmapFilePath)

#######################################################################################################################


def getDMapFilePath() -> str:
    """
    Returns the dmap file name which the library currently uses for looking up device(alias) names.
    """
    return pb.getDmapFile()

#######################################################################################################################
#######################################################################################################################


class GeneralRegisterAccessor(abc.ABC):
    """
    This is a super class to avoid code duplication. It contains
    methods that are common for the inheriting accessors.

    .. note:: As all accessors inherit from numpy's ndarray, the
              behaviour concerning slicing and mathematical operations
              is simimlar. Result accessors share the attributes of the left
              operand, hence they are shallow copies of it. This leads to
              functionality that is not available in the C++ implementation.
              Please refer to the examples below.

    Examples
    --------
    Slicing and writing. Operations are shared with the original accessor.
      >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
      >>> dev = da.Device("CARD_WITH_MODULES")
      >>> dev.open()
      >>> originAcc = dev.getTwoDRegisterAccessor(np.int32, "BOARD/DMA")
      >>> originAcc.set(7)
      >>> originAcc.write() # all elements are now 7.
      >>> print(originAcc)
          [[7 7 7 7 7 7]
          [7 7 7 7 7 7]
          [7 7 7 7 7 7]
          [7 7 7 7 7 7]]
      >>> channels = originAcc.getNChannels()
      >>> elementsPerChannel = originAcc.getNElementsPerChannel()
      >>> print(channels, elementsPerChannel) # there are 4 channels, each with 6 elements
          4 6
      >>> slicedAcc = originAcc[:][1] # the second element of every channel
      >>> slicedAcc.set(21) # set these to 21
      >>> slicedAcc.write()
      >>> print(originAcc) # originAcc is changed as well
          [[ 7  7  7  7  7  7]
           [21 21 21 21 21 21]
           [ 7  7  7  7  7  7]
           [ 7  7  7  7  7  7]]

    Results from mathematical operations are shallow copies of the left operand.
      >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
      >>> dev = da.Device("CARD_WITH_MODULES")
      >>> dev.open()
      >>> oneAcc = dev.getScalarRegisterAccessor(np.uint32, "ADC/WORD_CLK_CNT")
      >>> oneAcc.set(72)
      >>> oneAcc.write() # oneAcc is now 72
      >>> otherAcc = dev.getScalarRegisterAccessor(np.uint32, "ADC/WORD_CLK_CNT_1")
      >>> otherAcc.set(47)
      >>> otherAcc.write() #  otherAcc is now 47.
      >>> resultAcc = oneAcc + otherAcc # resultAcc's numpy buffer is now 119
      >>> print(resultAcc)
          [119]
      >>> resultAcc.write() # write() will also write into the register of oneAcc
      >>> oneAcc.read()
      >>> print(oneAcc)
          [119]
      >>> otherAcc.read() # the buffer's and registers of the right operand are not touched
      >>> print(otherAcc)
          [47]
      >>> resultAcc.getName() # the resultAcc is a shallow copy of the left operand
          '/ADC/WORD_CLK_CNT'
    """

    def read(self) -> None:
        """
        Read the data from the device.

        If :py:obj:`AccessMode.wait_for_new_data` was set, this function
        will block until new data has arrived. Otherwise, it still might block
        for a short time until the data transfer is complete.

        Examples
        --------
        Reading from a ScalarRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
          >>> acc.read()
          >>> acc
            ScalarRegisterAccessor([99], dtype=int32)

        """
        raise Exception('Not implemented in base class')

    def readLatest(self) -> bool:
        """
        Read the latest value, discarding any other update since the last read if present.

        Otherwise, this function is identical to :py:func:`readNonBlocking`,
        i.e. it will never wait for new values, and it will return
        whether a new value was available if
        :py:obj:`AccessMode.wait_for_new_data` is set.
        """
        raise Exception('Not implemented in base class')

    def readNonBlocking(self) -> bool:
        """
        Read the next value, if available in the input buffer.

        If :py:obj:`AccessMode.wait_for_new_data` was set, this function returns
        immediately and the return value indicated if a new value was
        available (`True`) or not (`False`).

        If :py:obj:`AccessMode.wait_for_new_data` was not set, this function is
        identical to :py:meth:`.read` , which will still return quickly. Depending on
        the actual transfer implementation, the backend might need to
        transfer data to obtain the current value before returning. Also
        this function is not guaranteed to be lock free. The return value
        will be always true in this mode.
        """
        raise Exception('Not implemented in base class')

    def write(self) -> bool:
        """
        Write the data to device.

        The return value is true, old data was lost on the write transfer
        (e.g. due to an buffer overflow). In case of an unbuffered write
        transfer, the return value will always be false.

        Examples
        --------
        Writing to a ScalarRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
          >>> reference = [199]
          >>> acc.set(reference)
          >>> acc.write()
            ScalarRegisterAccessor([199], dtype=int32)
        """
        raise Exception('Not implemented in base class')

    def writeDestructively(self) -> bool:
        """
        Just like :py:meth:`.write`, but allows the implementation
        to destroy the content of the user buffer in the process.

        The application must expect the user buffer of the
        TransferElement to contain undefined data after calling this function.
        """
        raise Exception('Not implemented in base class')

    def getName(self) -> str:
        """
        Returns the name that identifies the process variable.

        Examples
        --------
        Getting the name of a ScalarRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
          >>> acc.getName()
            '/ADC/WORD_CLK_CNT_1'

        """
        return self._accessor.getName()

    def getUnit(self) -> str:
        """
        Returns the engineering unit.

        If none was specified, it will default to "n./a."

        Examples
        --------
        Getting the engineering unit of a ScalarRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
          >>> acc.getUnit()
            'n./a.'

        """
        return self._accessor.getUnit()

    def getValueType(self) -> UserType:
        """
        Returns the type for the userType of this transfer element, that
        was given at the initialization of the accessor.

        Examples
        --------
        Getting the userType of a ScalarRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
          >>> acc.getValueType()
            numpy.int32

        """
        return self.userType

    def getDescription(self) -> str:
        """
        Returns the description of this variable/register, if there is any.

        Examples
        --------
        Getting the description of a ScalarRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
          >>> acc.getDescription()
            ''

        """
        return self._accessor.getDescription()

    def getAccessModeFlags(self) -> Sequence[AccessMode]:
        """
        Returns the access modes flags, that
        were given at the initialization of the accessor.

        Examples
        --------
        Getting the access modes flags of a OneDRegisterAccessor with the wait_for_new_data flag:
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getOneDRegisterAccessor(
            np.int32, "MODULE1/TEST_AREA_PUSH", 0, 0, [da.AccessMode.wait_for_new_data])
          >>> acc.getAccessModeFlags()
            [da.AccessMode.wait_for_new_data]

        """
        accessModeFlagStrings = self._accessor.getAccessModeFlagsString()
        flags = []
        for flag in accessModeFlagStrings.split(","):
            if flag == 'wait_for_new_data':
                flags.append(AccessMode.wait_for_new_data)
            if flag == 'raw':
                flags.append(AccessMode.raw)

        return flags

    def getVersionNumber(self) -> VersionNumber:
        """
        Returns the version number that is associated with the last transfer
        (i.e. last read or write). See :py:class:`VersionNumber` for details.

        Examples
        --------
        Getting the version number of a OneDRegisterAccessor:
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getOneDRegisterAccessor(
            np.int32, "MODULE1/TEST_AREA_PUSH", 0, 0, [da.AccessMode.wait_for_new_data])
          >>> acc.getVersionNumber()
            <_da_python_bindings.VersionNumber at 0x7f52b5f8a740>

        """
        return self._accessor.getVersionNumber()

    def isReadOnly(self) -> bool:
        """
        Check if transfer element is read only, i.e.
        it is readable but not writeable.

        Examples
        --------
        Getting the readOnly status of a OneDRegisterAccessor:
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getOneDRegisterAccessor(
            np.int32, "MODULE1/TEST_AREA_PUSH", 0, 0, [da.AccessMode.wait_for_new_data])
          >>> acc.isReadOnly()
            True

        """
        return self._accessor.isReadOnly()

    def isReadable(self) -> bool:
        """
        Check if transfer element is readable.

        It throws an exception if you try to read and :py:meth:`isReadable` is not True.

        Examples
        --------
        Getting the readable status of a OneDRegisterAccessor:
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getOneDRegisterAccessor(
            np.int32, "MODULE1/TEST_AREA_PUSH", 0, 0, [da.AccessMode.wait_for_new_data])
          >>> acc.isReadable()
            True

        """
        return self._accessor.isReadable()

    def isWriteable(self) -> bool:
        """
        Check if transfer element is writeable.

        It throws an exception if you try to write and :py:meth:`isWriteable` is not True.

        Examples
        --------
        Getting the writeable status of a OneDRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getOneDRegisterAccessor(
            np.int32, "MODULE1/TEST_AREA_PUSH", 0, 0, [da.AccessMode.wait_for_new_data])
          >>> acc.isReadable()
            False

        """
        return self._accessor.isWriteable()

    def isInitialised(self) -> bool:
        """
        Return if the accessor is properly initialized.

        It is initialized if it was constructed passing the
        pointer to an implementation, it is not
        initialized if it was constructed only using the placeholder
        constructor without arguments. Which should currently not happen,
        as the registerPath is a required argument for this module, but might
        be true for other implementations.

        Examples
        --------
        Getting the initialized status of a OneDRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getOneDRegisterAccessor(
              np.int32, "MODULE1/TEST_AREA_PUSH", 0, 0, [da.AccessMode.wait_for_new_data])
          >>> acc.isInitialised()
              True

        """
        return self._accessor.isInitialised()

    def setDataValidity(self, valid=DataValidity.ok) -> None:
        """
        Associate a persistent data storage object to be updated
        on each write operation of this ProcessArray.

        If no persistent data storage as associated previously, the
        value from the persistent storage is read and send to the receiver.

        .. note:: A call to this function will be ignored, if the
            TransferElement does not support persistent data storage
            (e.g. read-only variables or device registers)

        Parameters
        ----------
        valid: DataValidity
            DataValidity.ok or DataValidity.faulty

        """
        self._accessor.setDataValidity(valid)

    def dataValidity(self) -> DataValidity:
        """
        Return current validity of the data.

        Will always return :py:obj:`DataValidity.ok` if the backend does not support it

        """
        return self._accessor.dataValidity()

    def getId(self) -> TransferElementID:
        """
        Obtain unique ID for the actual implementation of this TransferElement.

        This means that e.g. two instances of ScalarRegisterAccessor
        created by the same call to :py:meth:`Device.getScalarRegisterAccessor`
        will have the same ID, while two instances obtained by to
        difference calls to :py:meth:`Device.getScalarRegisterAccessor`
        will have a different ID even when accessing the very same register.

        Examples
        --------
        Getting the name of a ScalarRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
          >>> acc.getId()
            <_da_python_bindings.TransferElementID at 0x7f5298a8f400>

        """
        return self._accessor.getId()

#######################################################################################################################
#######################################################################################################################


class NumpyGeneralRegisterAccessor(GeneralRegisterAccessor):
    def __init__(self, channels, elementsPerChannel, userType, accessor,
                 accessModeFlags: Sequence[AccessMode] = None) -> None:
        dtype = userType
        if dtype == str:
            dtype = 'U1'
        if channels is None:
            self.__array = np.zeros(shape=(elementsPerChannel), dtype=dtype)
        else:
            self.__array = np.zeros(shape=(channels, elementsPerChannel), dtype=dtype)
        self._accessor = accessor
        self.userType = userType
        self._AccessModeFlags = accessModeFlags

    def get(self) -> np.ndarray:
        return self.__array

    def set(self, value) -> None:
        """
        Set the user buffer to the given value.

        The value shape has to match the accessor, any mismatch will throw an exception.
        Different types will be converted to the userType of the accessor.

        Parameters
        ----------
        value : numpy.array or compatible type
            The new content of the user buffer.

        Examples
        --------
        Setting a ScalarRegisterAccessor
            >>> import deviceaccess as da
            >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
            >>> dev = da.Device("CARD_WITH_MODULES")
            >>> dev.open()
            >>> acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
            >>> acc.read()
                ScalarRegisterAccessor([74678], dtype=int32)
            >>> acc.set([-23])
            >>> acc.write()
                ScalarRegisterAccessor([-23], dtype=int32)

        Setting a OneDRegisterAccessor
            >>> import deviceaccess as da
            >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
            >>> dev = da.Device("CARD_WITH_MODULES")
            >>> dev.open()
            >>> acc = dev.getOneDRegisterAccessor(np.int32, "BOARD/WORD_CLK_MUX")
            >>> acc.read()
                OneDRegisterAccessor([342011132 958674678 342011132 958674678], dtype=int32)
            >>> acc.set([1, 9, 42, -23])
            >>> acc.write()
                OneDRegisterAccessor([  1,   9,  42, -23], dtype=int32)

        Setting a TwoDRegisterAccessor
            >>> import deviceaccess as da
            >>> dda.setDMapFilePath("deviceInformation/exampleCrate.dmap")
            >>> dev = da.Device("CARD_WITH_MODULES")
            >>> dev.open()
            >>> acc = dev.getTwoDRegisterAccessor(np.int32, "BOARD/DMA")
            >>> acc.read()
            TwoDRegisterAccessor([[  0   4  16  36  64 100]
                    [  0   0   0   0   0   0]
                    [  1   9  25  49  81 121]
                    [  0   0   0   0   0   0]], dtype=int32)
            >>> channels = acc.getNChannels()
            >>> elementsPerChannel = acc.getNElementsPerChannel()
            >>> reference = [
            >>>     [i*j+i+j+12 for j in range(elementsPerChannel)] for i in range(channels)]
            >>> acc.set(reference)
            >>> acc.write()
            TwoDRegisterAccessor([[12, 13, 14, 15, 16, 17],
                        [13, 15, 17, 19, 21, 23],
                        [14, 17, 20, 23, 26, 29],
                        [15, 19, 23, 27, 31, 35]], dtype=int32)

        """
        # TODO: maybe we should store scalars as such?
        if self.__array.dtype.kind != 'U':
            if not np.isscalar(value):
                newarray = np.asarray(value, dtype=self.__array.dtype)
            else:
                newarray = np.asarray([value], dtype=self.__array.dtype)
        else:
            if not np.isscalar(value):
                newarray = np.asarray(value)
            else:
                newarray = np.asarray([value])

        if self.__array.shape != newarray.shape:
            raise ValueError('The shape of the provided value is incompatible with this accessor.')
        self.__array = newarray

    # pass through all (non-operator) functions to the np array (unless defined here)
    def __getattr__(self, name):
        return self.__array.__getattribute__(name)

    # comparison operators
    def __lt__(self, other) -> bool:
        return self.__array < other

    def __le__(self, other) -> bool:
        return self.__array <= other

    def __gt__(self, other) -> bool:
        return self.__array > other

    def __ge__(self, other) -> bool:
        return self.__array >= other

    def __eq__(self, other) -> bool:
        return self.__array == other

    def __ne__(self, other) -> bool:
        return self.__array != other

    # conversion operators
    def __str__(self) -> str:
        return str(self.__array)

    def __bool__(self) -> bool:
        return bool(self.__array)

    # subscript operator
    def __getitem__(self, key) -> UserType:
        return self.__array[key]

    # binary operators
    def __add__(self, other) -> np.ndarray:
        return self.__array.__add__(other)

    def __sub__(self, other) -> np.ndarray:
        return self.__array.__sub__(other)

    def __mul__(self, other) -> np.ndarray:
        return self.__array.__mul__(other)

    def __truediv__(self, other) -> np.ndarray:
        return self.__array.__truediv__(other)

    def __floordiv__(self, other) -> np.ndarray:
        return self.__array.__floordiv__(other)

    def __mod__(self, other) -> np.ndarray:
        return self.__array.__mod__(other)

    def __pow__(self, other) -> np.ndarray:
        return self.__array.__pow__(other)

    def __rshift__(self, other) -> np.ndarray:
        return self.__array.__rshift__(other)

    def ___lshift___mul__(self, other) -> np.ndarray:
        return self.__array.__lshift__(other)

    def __and__(self, other) -> np.ndarray:
        return self.__array.__and__(other)

    def __or__(self, other) -> np.ndarray:
        return self.__array.__or__(other)

    def __xor__(self, other) -> np.ndarray:
        return self.__array.__xor__(other)

    # assignment operators
    def __isub__(self, other) -> RegisterAccessor:
        self.__array.__isub__(other)
        return self

    def __iadd__(self, other) -> RegisterAccessor:
        self.__array.__iadd__(other)
        return self

    def __imul__(self, other) -> RegisterAccessor:
        self.__array.__imul__(other)
        return self

    def __idiv__(self, other) -> RegisterAccessor:
        self.__array.__idiv__(other)
        return self

    def __ifloordiv__(self, other) -> RegisterAccessor:
        self.__array.__ifloordiv__(other)
        return self

    def __imod__(self, other) -> RegisterAccessor:
        self.__array.__imod__(other)
        return self

    def __ipow__(self, other) -> RegisterAccessor:
        self.__array.__ipow__(other)
        return self

    def __irshift__(self, other) -> RegisterAccessor:
        self.__array.__irshift__(other)
        return self

    def __ilshift__(self, other) -> RegisterAccessor:
        self.__array.__ilshift__(other)
        return self

    def __iand__(self, other) -> RegisterAccessor:
        self.__array.__iand__(other)
        return self

    def __ior__(self, other) -> RegisterAccessor:
        self.__array.__ior__(other)
        return self

    def __ixor__(self, other) -> RegisterAccessor:
        self.__array.__ixor__(other)
        return self

    # unary operators
    def __neg__(self) -> np.ndarray:
        return self.__array.__neg__()

    def __pos__(self) -> np.ndarray:
        return self.__array.__pos__()

    def __invert__(self) -> np.ndarray:
        return self.__array.__invert__()

    # accessor functions
    def read(self) -> None:
        self.__array = self._accessor.read(self.__array)

    def readNonBlocking(self) -> None:
        (status, self.__array) = self._accessor.readNonBlocking(self.__array)
        return status

    def readLatest(self) -> None:
        (status, self.__array) = self._accessor.readLatest(self.__array)
        return status

    def write(self) -> None:
        return self._accessor.write(self.__array)

    def writeDestructively(self) -> None:
        return self._accessor.writeDestructively(self.__array)

    # transfer doc strings from base class for accessor functions
    read.__doc__ = GeneralRegisterAccessor.read.__doc__
    readNonBlocking.__doc__ = GeneralRegisterAccessor.readNonBlocking.__doc__
    readLatest.__doc__ = GeneralRegisterAccessor.readLatest.__doc__
    write.__doc__ = GeneralRegisterAccessor.write.__doc__
    writeDestructively.__doc__ = GeneralRegisterAccessor.writeDestructively.__doc__

#######################################################################################################################
#######################################################################################################################


class TwoDRegisterAccessor(NumpyGeneralRegisterAccessor):
    """
    Accessor class to read and write registers transparently by using the accessor object
    like a 2D array of the type UserType.

    Conversion to and from the UserType will be handled by a data
    converter matching the register description in the map (if applicable).

    .. note:: As all accessors inherit from :py:obj:`GeneralRegisterAccessor`,
              please refer to the respective examples for the behaviour of
              mathematical operations and slicing with accessors.

    .. note:: Transfers between the device and the internal buffer need
            to be triggered using the read() and write() functions before reading
            from resp. after writing to the buffer using the operators.
    """

    def __init__(self, userType, accessor, accessModeFlags: Sequence[AccessMode] = None) -> None:
        channels = accessor.getNChannels()
        elementsPerChannel = accessor.getNElementsPerChannel()
        super().__init__(channels, elementsPerChannel, userType, accessor, accessModeFlags)

    def getNChannels(self) -> int:
        """
        Return number of channels.
        """
        return self._accessor.getNChannels()

    def getNElementsPerChannel(self) -> int:
        """
        Return number of elements/samples per channel.
        """
        return self._accessor.getNElementsPerChannel()

#######################################################################################################################
#######################################################################################################################


class OneDRegisterAccessor(NumpyGeneralRegisterAccessor):
    """
    Accessor class to read and write registers transparently by using the accessor object
    like a vector of the type UserType.

    Conversion to and from the UserType will be handled by a data
    converter matching the register description in the map (if applicable).

    .. note:: As all accessors inherit from :py:obj:`GeneralRegisterAccessor`,
              please refer to the respective examples for the behaviour of
              mathematical operations and slicing with accessors.

    .. note:: Transfers between the device and the internal buffer need
              to be triggered using the read() and write() functions before reading
              from resp. after writing to the buffer using the operators.
    """

    def __init__(self, userType, accessor, accessModeFlags: Sequence[AccessMode]) -> None:
        elements = accessor.getNElements()
        super().__init__(None, elements, userType, accessor, accessModeFlags)

    def getNElements(self) -> int:
        """
        Return number of elements/samples in the register.
        """
        return self._accessor.getNElements()

#######################################################################################################################
#######################################################################################################################


class ScalarRegisterAccessor(NumpyGeneralRegisterAccessor):
    """
    Accessor class to read and write scalar registers transparently by using the accessor object
    like a vector of the type UserType.

    Conversion to and from the UserType will be handled by a data
    converter matching the register description in the map (if applicable).

    .. note:: As all accessors inherit from :py:obj:`GeneralRegisterAccessor`,
              please refer to the respective examples for the behaviour of
              mathematical operations and slicing with accessors.

    .. note:: Transfers between the device and the internal buffer need
            to be triggered using the read() and write() functions before reading
            from resp. after writing to the buffer using the operators.
    """

    def __init__(self, userType, accessor, accessModeFlags: Sequence[AccessMode] = None) -> None:
        super().__init__(None, 1, userType, accessor, accessModeFlags)

    def readAndGet(self) -> np.number:
        """
        Convenience function to read and return a value of UserType.

        Examples
        --------
        Reading and Getting from a ScalarRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> dev.write("ADC/WORD_CLK_CNT_1", 37)
          >>> acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
          >>> acc.readAndGet()
            37

        """
        return self._accessor.readAndGet()

    def setAndWrite(self, newValue: np.number, versionNumber: VersionNumber = None) -> None:
        """
        Convenience function to set and write new value.

        Parameters
        ----------
        newValue : numpy.number and compatible types
          The content that should be written to the register.

        versionNumber: VersionNumber, optional
          The versionNumber that should be used for the write action.

        Examples
        --------
        Reading and Getting from a ScalarRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
          >>> acc.setAndWrite(38)
          >>> acc.readAndGet()
            38

        """
        if versionNumber is None:
            versionNumber = VersionNumber()
        self._accessor.setAndWrite(newValue, versionNumber)

    def writeIfDifferent(self, newValue: np.number, versionNumber: VersionNumber = None,
                         validity: DataValidity = DataValidity.ok) -> None:
        """
        Convenience function to set and write new value if it differes from the current value.

        The given version number is only used in case the value differs.

        Parameters
        ----------
        newValue : numpy.number and compatible types
          The contentthat should be written to the register.

        versionmNumber: VersionNumber, optional
          The versionNumber that should be used for the write action.

        Examples
        --------
        Reading and Getting from a ScalarRegisterAccessor
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getScalarRegisterAccessor(np.int32, "ADC/WORD_CLK_CNT_1")
          >>> acc.setAndWrite(38)
          >>> acc.writeIfDifferent(38) # will not write

        """
        if versionNumber is None:
            versionNumber = VersionNumber.getNullVersion()
        self._accessor.writeIfDifferent(newValue, versionNumber, validity)

#######################################################################################################################
#######################################################################################################################


class VoidRegisterAccessor(GeneralRegisterAccessor, np.ndarray):
    """
    Accessor class to read and write void registers transparently by using the accessor object..

    .. note:: Transfers between the device and the internal buffer need
            to be triggered using the read() and write() functions before reading
            from resp. after writing to the buffer using the operators.
    """
    def __new__(cls, accessor, accessModeFlags: Sequence[AccessMode] = None) -> None:
        obj = np.asarray(
            np.zeros(shape=(1, 1), dtype=np.void)).view(cls)
        obj = obj.ravel()
        obj._accessor = accessor
        obj._AccessModeFlags = accessModeFlags
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self._accessor = getattr(obj, '_accessor', None)
        self._AccessModeFlags = getattr(obj, '_AccessModeFlags', None)

    def read(self) -> None:
        self._accessor.read()

    def readLatest(self) -> bool:
        return self._accessor.readLatest()

    def readNonBlocking(self) -> bool:
        return self._accessor.readNonBlocking()

    def write(self) -> bool:
        return self._accessor.write()

    def writeDestructively(self) -> bool:
        return self._accessor.writeDestructively()

#######################################################################################################################
#######################################################################################################################


class Device:
    """ Construct Device from user provided device information

    This constructor is used to open a device listed in the dmap file.

    Parameters
    ----------
    aliasName : str
      The device alias/name in the dmap file for the hardware

    Examples
    --------
    Creating a device using a dmap file:
      >>> import deviceaccess as da
      >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
      # CARD_WITH_MODULES is an alias in the dmap file above
      >>> dev = da.Device("CARD_WITH_MODULES")
    Supports also with-statements:
      >>> with da.Device('CARD_WITH_MODULES') as dev:
      >>>     reg_value = dev.read('/PATH/TO/REGISTER')
    """
    # dict to get the corresponding function for each datatype
    _userTypeExtensions = {
        np.int8: "int8",
        np.uint8: "uint8",
        np.int16: "int16",
        np.uint16: "uint16",
        np.int32: "int32",
        np.uint32: "uint32",
        np.int64: "int64",
        np.uint64: "uint64",
        np.single: "float",
        np.float32: "float",
        np.double: "double",
        np.float64: "double",
        np.bool_: "boolean",
        bool: "boolean",
        str: "string",
    }

    def __init__(self, aliasName: str = None) -> None:
        self.aliasName = aliasName
        if aliasName:
            self._device = pb.getDevice(aliasName)
        else:
            self._device = pb.getDevice_no_alias()

    def __enter__(self) -> Device:
        """Helper function for with-statements"""
        if self.aliasName is None:
            raise SyntaxError('In a with-statement, an alias has to be provided in the device constructor!')
        else:
            self._device.open(self.aliasName)
            return self

    def __exit__(self, *args) -> None:
        """Helper function for with-statements"""
        self._device.close()

    def open(self, aliasName: str = None) -> None:
        """Open a :py:class:`Device`

        This method has to be called after the initialization to get accessors. It
        can also re-opens a :py:class:`Device` after :py:func:`close` was called.
        If no aliasName was giving during initialization, it is needed by
        this method.

        Parameters
        ----------
        aliasName : str, optional
          The :py:class:`Device` alias/name in the dmap file for the hardware

        Examples
        --------
        Opening a :py:class:`Device` without aliasName, as it has already been supplied at creation:
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()

        Opening a :py:class:`Device` with aliasName, as it has non been supplied at creation:
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device()
          >>> dev.open("CARD_WITH_MODULES")
        """
        if not aliasName:
            if self.aliasName:
                self._device.open()
            else:
                raise SyntaxError(
                    "No backend is assigned: the device is not opened"
                )
        elif not self.aliasName:
            self.aliasName = aliasName
            self._device.open(aliasName)
        else:
            raise SyntaxError(
                "Device has not been opened correctly: the device is not opened"
            )

    def close(self) -> None:
        """Close the :py:class:`Device`.

        The connection with the alias name is kept so the device can be re-opened
        using the :py:func:`open` function without argument.

        Examples
        --------
        Closing an open device:
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> dev.close()
        """
        self._device.close()

    def getCatalogueMetadata(self, metaTag: str) -> str:
        """Provides access to catalogue metadata.
        This method uses the metadata catalogue to access the value, defined at the beginning of a mapp file, i.e.:
        @TAG value
        By convention the tag is in uppercase.

        Parameters
        ----------
        metaTag
          Tag of the metadata, i.e.: MAPFILE_REVISION

        Returns
        -------
        String containing the metadata
        """
        return self._device.getCatalogueMetadata(metaTag)

    def getTwoDRegisterAccessor(
            self,
            userType,
            registerPathName: str,
            numberOfElements: int = 0,
            elementsOffset: int = 0,
            accessModeFlags: Sequence[AccessMode] = None) -> TwoDRegisterAccessor:
        """Get a :py:class:`TwoDRegisterAccessor` object for the given register.

        This allows to read and write transparently 2-dimensional registers.
        The optional arguments allow to restrict the accessor to a region of
        interest in the 2D register.

        Parameters
        ----------
        userType : type or numpy.dtype
          The userType for the accessor. Can be of any of the numpy.dtype
          combinations of float, int, 32 or 64-bit. Integers are also
          supported as signed, unsigned, 8 and 16-bit. E.g.
          `numpy.uint8`, or `numpy.float32`. There are also `str`, `bool`.

        registerPathName : str
          The name of the register to read from.

        numberOfElements : int, optional
          Specifies the number of elements per channel to read from the register.
          The width and fixed point representation of the register
          element are internally obtained from the map file.

          The method returns all elements in the register if this parameter is
          omitted or when its value is set as 0.

          If the value provided as this parameter exceeds the register size, an
          array with all elements upto the last element is returned.

        elementsOffset : int, optional
          This is a zero indexed offset from the first element of the register. When
          an elementIndexInRegister parameter is specified, the method reads out
          elements starting from this element index. The element at the index
          position is included in the read as well.

        accessModeFlags : list, optional
          A list to specify the access mode. It allows e.g. to enable raw access.
          See :py:class:`AccessMode` documentation for more details.
          Passing an access mode flag which is not supported by the backend or the given
          register will raise a NOT_IMPLEMENTED DeviceException.

        Examples
        --------
        Getting a Two-D Register Accessor of type uint8 from DMA; which is 6 elements long and has 4 channels:
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getTwoDRegisterAccessor(np.float32, "BOARD/DMA", 0, 0, [])
          >>> acc.read()
          >>> acc
          TwoDRegisterAccessor([[12., 13., 14., 15., 16., 17.],
                      [13., 15., 17., 19., 21., 23.],
                      [14., 17., 20., 23., 26., 29.],
                      [15., 19., 23., 27., 31., 35.]], dtype=float32)

        Getting a Two-D Register Accessor of type float64 from register "WORD_CLK_MUX" is 4 elements long.
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getTwoDRegisterAccessor(np.uint8, "BOARD/WORD_CLK_MUX")
          >>> acc.read()
          >>> acc
          TwoDRegisterAccessor([[42, 43, 44, 45]], dtype=uint8)
        """
        if not accessModeFlags:
            accessModeFlags = []
        # get function name according to userType
        userTypeFunctionExtension = self._userTypeExtensions.get(userType, None)
        if not userTypeFunctionExtension:
            raise SyntaxError("userType not supported")
        getTwoDAccessor = getattr(
            self._device, "getTwoDAccessor_" + userTypeFunctionExtension)

        accessor = getTwoDAccessor(
            registerPathName, numberOfElements, elementsOffset, accessModeFlags)
        twoDRegisterAccessor = TwoDRegisterAccessor(
            userType, accessor, accessModeFlags)
        return twoDRegisterAccessor

    def getOneDRegisterAccessor(
            self,
            userType,
            registerPathName: str,
            numberOfElements: int = 0,
            elementsOffset: int = 0,
            accessModeFlags: Sequence[AccessMode] = None) -> OneDRegisterAccessor:
        """Get a :py:class:`OneDRegisterAccessor` object for the given register.

        The OneDRegisterAccessor allows to read and write registers transparently by using
        the accessor object like a vector of the type UserType. If needed, the conversion
        to and from the UserType will be handled by a data converter matching the
        register description in e.g. a map file.

        Parameters
        ----------
        userType : type or numpy.dtype
          The userType for the accessor. Can be of any of the numpy.dtype
          combinations of float, int, 32 or 64-bit. Integers are also
          supported as signed, unsigned, 8 and 16-bit. E.g.
          `numpy.uint8`, or `numpy.float32`. There are also `str`, `bool`.

        registerPathName : str
          The name of the register to read from.

        numberOfElements : int, optional
          Specifies the number of elements per channel to read from the register.
          The width and fixed point representation of the register
          element are internally obtained from the map file.

          The method returns all elements in the register if this parameter is
          omitted or when its value is set as 0.

          If the value provided as this parameter exceeds the register size, an
          array with all elements upto the last element is returned.

        elementsOffset : int, optional
          This is a zero indexed offset from the first element of the register. When
          an elementIndexInRegister parameter is specified, the method reads out
          elements starting from this element index. The element at the index
          position is included in the read as well.

        accessModeFlags : list, optional
          A list to specify the access mode. It allows e.g. to enable raw access.
          See :py:class:`AccessMode` documentation for more details.
          Passing an access mode flag which is not supported by the backend or the given
          register will raise a NOT_IMPLEMENTED DeviceException.

        Examples
        --------
        Getting a One-D Register Accessor of type uint8 from WORD_STATUS; which is 1 element long:
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getOneDRegisterAccessor(np.uint8, "BOARD/WORD_STATUS")
          >>> acc.read()
          >>> acc
          OneDRegisterAccessor([255], dtype=uint8)

        Getting a One-D Register Accessor of type float64 from register "WORD_CLK_MUX" is 4 elements long.
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getOneDRegisterAccessor(np.uint8, "BOARD/WORD_CLK_MUX")
          >>> acc.read()
          >>> acc
          OneDRegisterAccessor([42., 43., 44., 45.], dtype=float32)
        """
        if not accessModeFlags:
            accessModeFlags = []
        # get function name according to userType
        userTypeFunctionExtension = self._userTypeExtensions.get(userType, None)
        if not userTypeFunctionExtension:
            raise SyntaxError("userType not supported")
        getOneDAccessor = getattr(
            self._device, "getOneDAccessor_" + userTypeFunctionExtension)

        accessor = getOneDAccessor(
            registerPathName, numberOfElements, elementsOffset, accessModeFlags)
        oneDRegisterAccessor = OneDRegisterAccessor(
            userType, accessor, accessModeFlags)
        return oneDRegisterAccessor

    def getScalarRegisterAccessor(
            self,
            userType,
            registerPathName: str,
            elementsOffset: int = 0,
            accessModeFlags: Sequence[AccessMode] = None) -> ScalarRegisterAccessor:
        """Get a :py:class:`ScalarRegisterAccessor` object for the given register.

        The ScalarRegisterObject allows to read and write registers transparently by using
        the accessor object like a variable of the type UserType. If needed, the conversion
        to and from the UserType will be handled by a data converter matching the register
        description in e.g. a map file.

        Parameters
        ----------
        userType : type or numpy.dtype
          The userType for the accessor. Can be of any of the numpy.dtype
          combinations of float, int, 32 or 64-bit. Integers are also
          supported as signed, unsigned, 8 and 16-bit. E.g.
          `numpy.uint8`, or `numpy.float32`. There are also `str`, `bool`.

        registerPathName : str
          The name of the register to read from.

        elementsOffset : int, optional
          This is a zero indexed offset from the first element of the register. When
          an elementIndexInRegister parameter is specified, the method reads out
          elements starting from this element index. The element at the index
          position is included in the read as well.

        accessModeFlags : list, optional
          A list to specify the access mode. It allows e.g. to enable raw access.
          See :py:class:`AccessMode` documentation for more details.
          Passing an access mode flag which is not supported by the backend or the given
          register will raise a NOT_IMPLEMENTED DeviceException.

        Examples
        --------
        Getting a scalar Register Accessor of type int16 from WORD_STATUS:
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getScalarRegisterAccessor(np.int16, "ADC/WORD_STATUS")
          >>> acc.read()
          >>> acc
          ScalarRegisterAccessor([32767], dtype=int16)

        """
        if not accessModeFlags:
            accessModeFlags = []
        # get function name according to userType
        userTypeFunctionExtension = self._userTypeExtensions.get(userType, None)
        if not userTypeFunctionExtension:
            raise SyntaxError("userType not supported")
        getScalarAccessor = getattr(
            self._device, "getScalarAccessor_" + userTypeFunctionExtension)

        accessor = getScalarAccessor(
            registerPathName, elementsOffset, accessModeFlags)
        scalarRegisterAccessor = ScalarRegisterAccessor(
            userType, accessor, accessModeFlags)
        return scalarRegisterAccessor

    def getVoidRegisterAccessor(self, registerPathName: str, accessModeFlags: Sequence[AccessMode] = None) -> VoidRegisterAccessor:
        """Get a :py:class:`VoidRegisterAccessor` object for the given register.

        The VoidRegisterAccessor allows to read and write registers. Getting a read
        accessor is only possible with the wait_for_new_data flag. This access mode
        will be rejected for write accessors.

        Parameters
        ----------
        registerPathName : str
          The name of the register to read from.

        accessModeFlags : list, optional
          A list to specify the access mode. It allows e.g. to enable wait_for_new_data access.
          See :py:class:`AccessMode` documentation for more details.
          Passing an access mode flag which is not supported by the backend or the given
          register will raise a NOT_IMPLEMENTED DeviceException.

        Examples
        --------
        Sending interrupts per Void Accessor:
          >>> import deviceaccess as da
          >>> da.setDMapFilePath("deviceInformation/push.dmap")
          >>> dev = da.Device("SHARED_RAW_DEVICE")
          >>> dev.open()
          >>> dev.activateAsyncRead()
          >>>
          >>> writeAcc = dev.getOneDRegisterAccessor(np.int32, "MODULE1/TEST_AREA")
          >>> arr1to10 = np.array(range(1, 11), dtype=np.int32)
          >>> writeAcc.set(arr1to10)
          >>> writeAcc.write()
          >>>
          >>> readAcc = dev.getOneDRegisterAccessor(
          >>>     np.int32, "MODULE1/TEST_AREA_PUSH", 0, 0, [da.AccessMode.wait_for_new_data])
          >>> readAcc.read() # first read is always non-blocking
            OneDRegisterAccessor([ 1  2  3  4  5  6  7  8  9 10]], dtype=int32)
          >>> # double values of writeAccReg
          >>> writeAcc += arr1to10
          >>> writeAcc.write()
          >>> interruptAcc = dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_2")
          >>> interruptAcc.write() # interrupt needed, otherwise second read would be blocking
          >>>
          >>> readAcc.read()
            OneDRegisterAccessor(
                [ 2,  4,  6,  8, 10, 12, 14, 16, 18, 20], dtype=int32)

        """
        if not accessModeFlags:
            accessModeFlags = []
        accessor = self._device.getVoidAccessor(
            registerPathName, accessModeFlags)
        voidRegisterAccessor = VoidRegisterAccessor(accessor, accessModeFlags)
        return voidRegisterAccessor

    def activateAsyncRead(self) -> None:
        """
        Activate asynchronous read for all transfer elements where
        :py:obj:`AccessMode.wait_for_new_data` is set.

        If this method is called while the device is not opened or has an error,
        this call has no effect. If it is called when no deactivated transfer
        element exists, this call also has no effect.
        On return, it is not guaranteed that
        all initial values have been received already.

        See also
        --------
        :py:func:`getVoidRegisterAccessor`: has a usage example.
        """
        self._device.activateAsyncRead()

    def getRegisterCatalogue(self) -> pb.RegisterCatalogue:
        """
        Return the register catalogue with detailed information on all registers.
        """
        return self._device.getRegisterCatalogue()

    def read(self, registerPath: str, dtype: np.dtype = np.float64, numberOfWords: int = 0,
             wordOffsetInRegister: int = 0, accessModeFlags: Sequence[AccessMode] = None) -> np.ndarray | np.number:
        """
        Inefficient convenience function to read a register without obtaining an accessor.
        If no dtype is selected, the returned ndarray will default to np.float64.
        If numberOfWords is not specified, it takes the maximm minus the offset.
        If numberOfChannels is not specified, it takes the maximm possible.
        """
        accessModeFlags = [] if accessModeFlags is None else accessModeFlags
        arr = self._device.read(registerPath, numberOfWords, wordOffsetInRegister, accessModeFlags)
        if arr.shape == (1, 1):
            return arr[0][0]
        if arr.shape[0] == 1:
            return arr[:][0]
        return arr

    def write(
            self,
            registerPath: str,
            dataToWrite: np.ndarray | np.number,
            wordOffsetInRegister: int = 0,
            accessModeFlags: Sequence[AccessMode] = None) -> None:
        """
        Inefficient convenience function to write a register without obtaining an accessor.
        If no dtype is selected, the returned ndarray will default to np.float64.
        """

        # make proper array, if number was submitted
        if isinstance(dataToWrite, list):
            array = np.array(dataToWrite)
            # upgrade 1d-list input two the 2d that is expected for scalar and 1d lists:
            if not array.ndim == 2:
                array = np.array([dataToWrite])
        elif not isinstance(dataToWrite, np.ndarray):
            array = np.array([[dataToWrite]])
        else:
            array = dataToWrite
        numberOfElements = array.shape[1] if array.ndim == 2 else (array.shape[0] if array.ndim == 1 else 1)

        accessModeFlags = [] if accessModeFlags is None else accessModeFlags
        self._device.write(array, registerPath, numberOfElements,
                           wordOffsetInRegister, accessModeFlags)


#######################################################################################################################
# Type Definitions


UserType = functools.reduce(lambda x, y: Union[x, y], list(Device._userTypeExtensions.values()))

RegisterAccessor = Union[OneDRegisterAccessor,
                         TwoDRegisterAccessor,
                         ScalarRegisterAccessor]

import _da_python_bindings as pb
import numpy as np
import enum
from _da_python_bindings import AccessMode, DataValidity, TransferElementID, VersionNumber
import abc


def setDMapFilePath(dmapFilePath):
    # dmapFilePath	Relative or absolute path of the dmap file (directory and file name).
    pb.setDmapFile(dmapFilePath)


def getDMapFilePath(dmapFilePath):
    return pb.getDmapFile()


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
      >>> dev = da.Device("CARD_WITH_MODULES") # CARD_WITH_MODULES is an alias in the dmap file above

    """
    # dict to get the corresponding function for each datatype
    _userTypeExtensions = {
        np.int32: "int32",
        np.int16: "int16",
        np.int8: "int8",
        np.uint8: "uint8",
        np.int16: "int16",
        np.uint16: "uint16",
        np.int32: "int32",
        np.uint32: "uint32",
        np.int64: "int64",
        np.uint64: "uint64",
        float: "float",
        np.float: "float",
        np.float32: "float",
        np.double: "double",
        np.float64: "double",
        np.str_: "string",
        np.bool: "boolean"
    }

    def __init__(self, aliasName=None):
        self.aliasName = aliasName
        if aliasName:
            self._device = pb.getDevice(aliasName)
        else:
            self._device = pb.getDevice_no_alias()

    def open(self, aliasName=None):
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

    def close(self):
        """Close the :py:class:`Device`.

        The connection with the alias name is kept so the device can be re-opened using the :py:func:`open` function without argument. 

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

    def getTwoDRegisterAccessor(self, userType, registerPathName, numberOfElements=0, elementsOffset=0, accessModeFlags=[]):
        """Get a :py:class:`TwoDRegisterAccessor` object for the given register. 

        This allows to read and write transparently 2-dimensional registers. 
        The optional arguments allow to restrict the accessor to a region of 
        interest in the 2D register.

        Parameters
        ----------
        userType : type or numpy.dtype
          The userType for the accessor. Can be float, or any of the numpy.dtype 
          combinations of signed, unsigned, double, int, 8, 16, 32 or 64-bit. E.g. 
          `numpy.uint8`, or `numpy.float32`.

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
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getTwoDRegisterAccessor(np.uint8, "BOARD/WORD_CLK_MUX")
          >>> acc.read()
          >>> acc
          TwoDRegisterAccessor([[42, 43, 44, 45]], dtype=uint8)
        """
        # get function name according to userType
        userTypeFunctionExtension = self._userTypeExtensions.get(
            userType, None)
        if not userTypeFunctionExtension:
            raise SyntaxError(
                "userType not supported"
            )
        getTwoDAccessor = getattr(
            self._device, "getTwoDAccessor_" + userTypeFunctionExtension)

        accessor = getTwoDAccessor(
            registerPathName, numberOfElements, elementsOffset, accessModeFlags)
        twoDRegisterAccessor = TwoDRegisterAccessor(
            userType, accessor, accessModeFlags)
        return twoDRegisterAccessor

    def getOneDRegisterAccessor(self, userType, registerPathName, numberOfElements=0, elementsOffset=0, accessModeFlags=[]):
        """Get a :py:class:`OneDRegisterAccessor` object for the given register. 

        The OneDRegisterAccessor allows to read and write registers transparently by using 
        the accessor object like a vector of the type UserType. If needed, the conversion 
        to and from the UserType will be handled by a data converter matching the 
        register description in e.g. a map file.

        Parameters
        ----------
        userType : type or numpy.dtype
          The userType for the accessor. Can be float, or any of the numpy.dtype 
          combinations of signed, unsigned, double, int, 8, 16, 32 or 64-bit. E.g. 
          `numpy.uint8`, or `numpy.float32`.

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
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getOneDRegisterAccessor(np.uint8, "BOARD/WORD_STATUS")
          >>> acc.read()
          >>> acc
          OneDRegisterAccessor([255], dtype=uint8)

        Getting a One-D Register Accessor of type float64 from register "WORD_CLK_MUX" is 4 elements long.
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getOneDRegisterAccessor(np.uint8, "BOARD/WORD_CLK_MUX")
          >>> acc.read()
          >>> acc
          OneDRegisterAccessor([42., 43., 44., 45.], dtype=float32)
        """
        # get function name according to userType
        userTypeFunctionExtension = self._userTypeExtensions.get(
            userType, None)
        if not userTypeFunctionExtension:
            raise SyntaxError(
                "userType not supported"
            )
        getOneDAccessor = getattr(
            self._device, "getOneDAccessor_" + userTypeFunctionExtension)

        accessor = getOneDAccessor(
            registerPathName, numberOfElements, elementsOffset, accessModeFlags)
        oneDRegisterAccessor = OneDRegisterAccessor(
            userType, accessor, accessModeFlags)
        return oneDRegisterAccessor

    def getScalarRegisterAccessor(self, userType, registerPathName, elementsOffset=0, accessModeFlags=[]):
        """Get a :py:class:`ScalarRegisterAccessor` object for the given register. 

        The ScalarRegisterObject allows to read and write registers transparently by using
        the accessor object like a variable of the type UserType. If needed, the conversion 
        to and from the UserType will be handled by a data converter matching the register 
        description in e.g. a map file.

        Parameters
        ----------
        userType : type or numpy.dtype
          The userType for the accessor. Can be float, or any of the numpy.dtype 
          combinations of signed, unsigned, double, int, 8, 16, 32 or 64-bit. E.g. 
          `numpy.uint8`, or `numpy.float32`.

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
          >>> da.setDMapFilePath("deviceInformation/exampleCrate.dmap")
          >>> dev = da.Device("CARD_WITH_MODULES")
          >>> dev.open()
          >>> acc = dev.getScalarRegisterAccessor(np.int16, "ADC/WORD_STATUS")
          >>> acc.read()
          >>> acc
          ScalarRegisterAccessor([32767], dtype=int16)

        """
        # get function name according to userType
        userTypeFunctionExtension = self._userTypeExtensions.get(
            userType, None)
        if not userTypeFunctionExtension:
            raise SyntaxError(
                "userType not supported"
            )
        getScalarAccessor = getattr(
            self._device, "getScalarAccessor_" + userTypeFunctionExtension)

        accessor = getScalarAccessor(
            registerPathName, elementsOffset, accessModeFlags)
        scalarRegisterAccessor = ScalarRegisterAccessor(
            userType, accessor, accessModeFlags)
        return scalarRegisterAccessor

    def getVoidRegisterAccessor(self, registerPathName, accessModeFlags=[]):
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
          >>> da.setDMapFilePath("deviceInformation/push.dmap")
          >>> dev = da.Device("SHARED_RAW_DEVICE")
          >>> dev.open()
          >>> dev.activateAsyncRead()
          >>> 
          >>> writeAcc = dev.getOneDRegisterAccessor(np.int32, "MODULE1/TEST_AREA")
          >>> arr1to10 = np.array([i for i in range(1, 11)], dtype=np.int32)
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
          >>> interruptAcc = dev.getVoidRegisterAccessor("DUMMY_INTERRUPT_2_3")
          >>> interruptAcc.write() # interrupt needed, otherwise second read would be blocking
          >>> 
          >>> readAcc.read()
            OneDRegisterAccessor([ 2,  4,  6,  8, 10, 12, 14, 16, 18, 20], dtype=int32)

        """
        accessor = self._device.getVoidAccessor(
            registerPathName, accessModeFlags)
        voidRegisterAccessor = VoidRegisterAccessor(accessor, accessModeFlags)
        return voidRegisterAccessor

    def activateAsyncRead(self):
        """
        Activate asynchronous read for all transfer elements where 
        :py:object:`AccessMode.wait_for_new_data` is set.

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


class GeneralRegisterAccessor:

    def read(self):
        self._accessor.read(self.view())

    def readLatest(self):
        return self._accessor.readLatest(self.view())

    def readNonBlocking(self):
        return self._accessor.readNonBlocking(self.view())

    def write(self):
        return self._accessor.write(self.view())

    def writeDestructively(self):
        return self._accessor.writeDestructively(self.view())

    def getName(self):
        return self._accessor.getName()

    def getUnit(self):
        return self._accessor.getUnit()

    def getValueType(self):
        return self.userType

    def getDescription(self):
        return self._accessor.getDescription()

    def getAccessModeFlags(self):
        return self._AccessModeFlags

    def getVersionNumber(self):
        return self._accessor.getVersionNumber()

    def isReadOnly(self):
        return self._accessor.isReadOnly()

    def isReadable(self):
        return self._accessor.isReadable()

    def isWriteable(self):
        return self._accessor.isWriteable()

    def isInitialised(self):
        return self._accessor.isInitialised()

    def setDataValidity(self, valid=DataValidity.ok):
        self._accessor.setDataValidity(valid)

    def dataValidity(self):
        return self._accessor.dataValidity()

    def getId(self):
        return self._accessor.getId()


class TwoDRegisterAccessor(GeneralRegisterAccessor, np.ndarray):

    def __new__(self, userType, accessor, accessModeFlags=None):
        # add the new attribute to the created instance
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        channels = accessor.getNChannels()
        elementsPerChannel = accessor.getNElementsPerChannel()
        obj = np.asarray(
            np.zeros(shape=(channels, elementsPerChannel), dtype=userType)).view(self)
        obj._accessor = accessor
        obj.userType = userType
        obj._AccessModeFlags = accessModeFlags
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self._accessor = getattr(obj, '_accessor', None)
        self.userType = getattr(obj, 'userType', None)
        self._AccessModeFlags = getattr(obj, '_AccessModeFlags', None)

    def getNChannels(self):
        return self._accessor.getNChannels()

    def getNElementsPerChannel(self):
        return self._accessor.getNElementsPerChannel()

    def set(self, array):
        self *= 0
        self += array


class OneDRegisterAccessor(GeneralRegisterAccessor, np.ndarray):

    def __new__(cls, userType, accessor, accessModeFlags=None):
        elements = accessor.getNElements()
        obj = np.asarray(
            np.zeros(shape=(elements), dtype=userType)).view(cls)
        accessor.linkUserBufferToNpArray(obj)
        # obj was 2d beforehand, ravel does not copy compared to flatten.
        #obj = obj
        obj._accessor = accessor
        obj.userType = userType
        obj._AccessModeFlags = accessModeFlags
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self._accessor = getattr(obj, '_accessor', None)
        self.userType = getattr(obj, 'userType', None)
        self._AccessModeFlags = getattr(obj, '_AccessModeFlags', None)

    """ remove to get zero-copy
    def read(self):
        self._accessor.read()

    def readLatest(self):
        return self._accessor.readLatest()

    def readNonBlocking(self):
        return self._accessor.readNonBlocking()

    def write(self):
        return self._accessor.write()

    def writeDestructively(self):
        return self._accessor.writeDestructively()
        """

    def getNElements(self):
        return self._accessor.getNElements()

    def set(self, array):
        self *= 0
        self += array


class ScalarRegisterAccessor(GeneralRegisterAccessor, np.ndarray):

    def __new__(cls, userType, accessor, accessModeFlags=None):
        elements = 1
        obj = np.asarray(
            np.zeros(shape=(elements), dtype=userType)).view(cls)
        # obj was 2d beforehand, ravel does not copy compared to flatten.
        #obj = obj
        obj._accessor = accessor
        obj.userType = userType
        obj._AccessModeFlags = accessModeFlags
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self._accessor = getattr(obj, '_accessor', None)
        self.userType = getattr(obj, 'userType', None)
        self._AccessModeFlags = getattr(obj, '_AccessModeFlags', None)

    def __lt__(self, other):
        return self[0] < other

    def set(self, scalar):
        self *= 0
        self += scalar


class VoidRegisterAccessor(GeneralRegisterAccessor, np.ndarray):

    def __new__(cls, accessor, accessModeFlags=None):
        obj = np.asarray(
            np.zeros(shape=(1, 1), dtype=np.void)).view(cls)
        obj = obj.ravel()
        obj._accessor = accessor
        obj._AccessModeFlags = accessModeFlags
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self._accessor = getattr(obj, '_accessor', None)
        self._AccessModeFlags = getattr(obj, '_AccessModeFlags', None)

    def read(self):
        self._accessor.read()

    def readLatest(self):
        return self._accessor.readLatest()

    def readNonBlocking(self):
        return self._accessor.readNonBlocking()

    def write(self):
        return self._accessor.write()

    def writeDestructively(self):
        return self._accessor.writeDestructively()

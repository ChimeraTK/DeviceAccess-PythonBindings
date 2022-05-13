import _da_python_bindings as pb
import numpy as np
import enum
from _da_python_bindings import AccessMode, DataValidity, TransferElementID, VersionNumber


def setDMapFilePath(dmapFilePath):
    # dmapFilePath	Relative or absolute path of the dmap file (directory and file name).
    pb.setDmapFile(dmapFilePath)


def getDMapFilePath(dmapFilePath):
    return pb.getDmapFile()


class Device:
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
        np.float: "float",
        np.double: "double",
        np.string_: "string",
        np.bool: "boolean"
    }

    def __init__(self, aliasName=None):
        self.aliasName = aliasName
        if aliasName:
            self._device = pb.getDevice(aliasName)
        else:
            self._device = pb.getDevice_no_alias()

    def open(self, aliasName=None):
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
        self._device.close()

    def getTwoDRegisterAccessor(self, userType, registerPathName, numberOfElements=0, elementsOffset=0, accessModeFlags=[]):

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
        twoDRegisterAccessor = TwoDRegisterAccessor(userType, accessor)
        return twoDRegisterAccessor

    def getOneDRegisterAccessor(self, userType, registerPathName, numberOfElements=0, elementsOffset=0, accessModeFlags=[]):

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
        oneDRegisterAccessor = OneDRegisterAccessor(userType, accessor)
        return oneDRegisterAccessor


class GeneralRegisterAccessor(np.ndarray):
    def __new__(cls, userType, accessor, accessModeFlags=None):
        pass

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

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


class TwoDRegisterAccessor(GeneralRegisterAccessor):

    def __new__(cls, userType, accessor, accessModeFlags=None):
        # add the new attribute to the created instance
        cls._accessor = accessor
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        channels = accessor.getNChannels()
        elementsPerChannel = accessor.getNElementsPerChannel()
        cls.userType = userType
        cls._AccessModeFlags = accessModeFlags
        obj = np.asarray(
            np.zeros(shape=(channels, elementsPerChannel), dtype=userType)).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def getNChannels(self):
        return self._accessor.getNChannels()

    def getNElementsPerChannel(self):
        return self._accessor.getNElementsPerChannel()


class OneDRegisterAccessor(GeneralRegisterAccessor):

    def __new__(cls, userType, accessor, accessModeFlags=None):
        # add the new attribute to the created instance
        cls._accessor = accessor
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        elements = accessor.getNElements()
        cls.userType = userType
        cls._AccessModeFlags = accessModeFlags
        obj = np.asarray(
            np.zeros(shape=(1, elements), dtype=userType)).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def getNElements(self):
        return self._accessor.getNElements()

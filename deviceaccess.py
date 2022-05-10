import _da_python_bindings as pb
import numpy as np
import enum
from _da_python_bindings import AccessMode, DataValidity


def setDMapFilePath(dmapFilePath):
    # dmapFilePath	Relative or absolute path of the dmap file (directory and file name).
    pb.setDmapFile(dmapFilePath)


def getDMapFilePath(dmapFilePath):
    return pb.getDmapFile()


class Device:
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


class TwoDRegisterAccessor(np.ndarray):

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

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)

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

    def getNChannels(self):
        return self._accessor.getNChannels()

    def getNElementsPerChannel(self):
        return self._accessor.getNElementsPerChannel()

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
        # TODO
        print("Not yet implemented")
        pass

    def isReadOnly(self):
        return self._accessor.isReadOnly()

    def isReadable(self):
        return self._accessor.isReadable()

    def isWriteable(self):
        return self._accessor.isWriteable()

    def isInitialised(self):
        return self._accessor.isInitialised()

    def getId(self):
        # TODO
        print("Not yet implemented")
        pass

    def setDataValidity(self, valid=DataValidity.ok):
        self._accessor.setDataValidity(valid)

    def dataValidity(self):
        return self._accessor.dataValidity()

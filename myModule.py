import mtcamappeddevice 
import numpy


class Device():

    def __init__(self, deviceName, mapFile):
        # exception handling is a kludge for now
        try:
            mtcamappeddevice.createDevice(5)
        except Exception as self.__storedArgErrException:
            pass

        try:
                self.__openedDevice = mtcamappeddevice.createDevice(deviceName)
        except Exception, e:            
            if(e.__class__ == self.__storedArgErrException.__class__):
                print "Device name and mapfile name are expected to be strings"

    def read(self, registerName, numberOf32bitWordsToRead=1,
            offsetFromRegisterBaseAddress=0):
""" string, integer, integer -> list of double values
        This is an example of a module level function.

    Function parameters should be documented in the ``Parameters`` section.
    The name of each parameter is required. The type and description of each
    parameter is optional, but should be included if not obvious.

    If the parameter itself is optional, it should be noted by adding
    ", optional" to the type. If \*args or \*\*kwargs are accepted, they
    should be listed as \*args and \*\*kwargs.

    The format for a parameter is::

        name : type
            description

            The description may span multiple lines. Following lines
            should be indented to match the first line of the description.

            Multiple paragraphs are supported in parameter
            descriptions.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional
        The second parameter, defaults to None.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    bool
        True if successful, False otherwise.

        The return type is not optional. The ``Returns`` section may span
        multiple lines and paragraphs. Following lines should be indented to
        match the first line of the description.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises
    ------
    AttributeError
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.
    ValueError
        If `param2` is equal to `param1`.

    

                device = self.__openedDevice


        # find the size of array in case of mapped:
        #array = numpy.zeros(numberOfWords, dtype = numpy.int32)
        #numberOfWords = numberOfWords * 4
        #if(type(addressOffset) != int):
            #print "offset is not type int"
            #return None
#
        #if(type(registerName) == str):
            #device.readRaw(registerName, array, numberOfWords*4, addressOffset)
        #else:
            #device.readRaw(addressOffset, array, numberOfWords*4, bar)
        #return array

    def readRaw(self, registerName, numberOf32BitWordsToRead=1
            offsetFromRegisterBaseAddress=0):


    def write(self, registerName, dataToWrite, offsetFromRegisterBaseAddress=0):
        pass

    def writeRaw(self, registerName, dataToWrite,
            offsetFromRegisterBaseAddress=0):
        #device = self.__openedDevice
        #wordsToWrite = array.size * 4
        #if(wordsToWrite == 0 ):
            #print "Nothing to write"
            #return None

        #if(type(addressOffset) != int):
            #print "offset is not type int"
            #return None

        #if(array.dtype != numpy.int32):
            #print "expecting a numpy.int32 type array"
            #return None

        #if(type(registerName) == str):
            #device.writeRaw(registerName, array, wordsToWrite, addressOffset)
        #else:
            #device.writeRaw(addressOffset, array, wordsToWrite, bar)



        
    def wrapper(self, regoffset=0, numpyArray=numpy.array([1], dtype=numpy.int32) ,
            size=1, bar=0):
        print self.__openedDevice
        #device = Device()
        #device.readRaw(regoffset, numpyArray, size, bar)


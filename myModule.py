import mtcamappeddevice 
import numpy


class Device():
  """ This class represents the hardware device to access 
  
  This class can be used to open and acess the registers of a mapped device
  
  Parameters
  ----------
  deviceName : str
    The device file identifier for the hardware

  mapFile : str
    The location of the register mapped file for the hardware under
    consideration
  
  Examples
  --------
  >>> device = Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
  """
  
  def __init__(self, deviceName, mapFile):
    """ Constructor for the Device class
    """
  #exception handling is a kludge for now
    try:
      mtcamappeddevice.createDevice(5)
    except Exception as self.__storedArgErrException:
      pass

    try:
      self.__openedDevice = mtcamappeddevice.createDevice(deviceName)
    except Exception, e:            
      if(e.__class__ == self.__storedArgErrException.__class__):
        print "Device name and mapfile name are expected to be strings"

  def read(self, registerName, numberOfElementsToRead=1,
            offsetFromRegisterBaseAddress=0):
    """ Reads out Fixed point converted values from the opened mapped device
    
    This method uses the register mapping information to return Fixed Point
    converted version of the values contained in a register. This method can be
    used to read in the whole register or an arbitary number of register
    elements from anywhere in the span of the register (through the
    'offsetFromRegisterBaseAddress' parameter).
    
    Parameters
    ----------
    registerName : str
      The name of the register on the device to which read access is sought.
      
    numberOfElementsToRead : int, optional 
      Optional parameter specifying the number of register elements that should
      be read out. The width of each register element is internally obtained
      from the mapping file.
      
      The method returns all elements in the register if this parameter is
      ommitted or when its value is set as 0.
      
    offsetFromRegisterBaseAddress : int, optional
      This is an offset from the start of the specified register's base address.
      An offset of 1 represents 32 bits. When an offset is provided as a
      parameter, the method returns elements from this point in memory (/address
      offset) onwards.
      
    Returns
    -------
    numpy.ndarray
      The return type for the method is a 1-Dimensional array with datatype
      numpy.float32. The returned 1-Dimensional numpy.ndarray would either
      contain all elements in the register or only the number specified by the
      numberOfElementsToRead parameter
     
    Examples
    --------
    >>> device = Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
    
    >>> device.read("WORD_CLK_MUX")
    array([15.0,14.0, 13.0, 12.0], dtype = float32)#TODO fill in the output
    
    >>> device.read("WORD_CLK_MUX", 0)
    array([15.0,14.0, 13.0, 12.0], dtype = float32)#TODO fill in the output
        
    >>> device.read("WORD_CLK_CNT", 1)
    array([15.0], dtype = float32)#TODO fill in the output
    
    >>> device.read("WORD_CLK_CNT", 1, 2 )
    array([13.0], dtype = float32)

    >>> device.read("WORD_CLK_CNT", 0, 2 )
    array([13.0, 12.0], dtype = float32)
        
    >>> device.read("WORD_CLK_CNT", numberOfElementsToRead=1, offsetFromRegisterBaseAddress=2 )
    array([13.0], dtype = float32)
    
    >>> device.read("WORD_CLK_CNT", offsetFromRegisterBaseAddress=2 )
    array([13.0, 12.0], dtype = float32)
    """
    return numpy.array([0], dtype = numpy.float32)

  def readRaw(self, registerName, numberOf32BitWordsToRead=1, 
              offsetFromRegisterBaseAddress=0):
    pass
  
  def write(self, registerName, dataToWrite, offsetFromRegisterBaseAddress=0):
    pass

  def writeRaw(self, registerName, dataToWrite,
      offsetFromRegisterBaseAddress=0):
    pass
  
  def wrapper(self, regoffset=0, numpyArray=numpy.array([1], dtype=numpy.int32),
               size=1, bar=0):
    pass

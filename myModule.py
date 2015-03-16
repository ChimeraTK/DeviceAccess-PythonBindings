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
    converted versions of the values contained in a register. This method can be
    used to read in the whole register or an arbitary number of register
    elements from anywhere within the register (through the
    'offsetFromRegisterBaseAddress' parameter).
    
    Parameters
    ----------
    registerName : str
      The name of the register (on the device) to which read access is sought.
      
    numberOfElementsToRead : int, optional 
      Optional parameter specifying the number of register elements that should
      be read out. The width and fixed point representation of the register
      element are internally obtained from the mapping file.
      
      The method returns all elements in the register if this parameter is
      ommitted or when its value is set as 0.
      
    offsetFromRegisterBaseAddress : int, optional
      This is an offset from the start of the specified register's base address.
      An offset of 1 represents 32 bits. When an offset is provided as a
      parameter, the method reads out elements from this point in memory
      (/address offset) onwards.
      
    Returns
    -------
    numpy.array, dtype == numpy.float32
      The return type for the method is a 1-Dimensional numpy array with
      datatype numpy.float32. The returned numpy.array would either contain all
      elements in the register or only the number specified by the
      numberOfElementsToRead parameter
     
    Raises
    ------
    
    Examples
    --------
    In the examples  register "WORD_CLK_MUX" is 4 elements long.
    
    >>> device = Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
    
    >>> device.read("WORD_CLK_MUX")
    array([15.0, 14.0, 13.0, 12.0], dtype = float32)#TODO fill in the output
    
    >>> device.read("WORD_CLK_MUX", 0)
    array([15.0, 14.0, 13.0, 12.0], dtype = float32)#TODO fill in the output
        
    >>> device.read("WORD_CLK_MUX", 1)
    array([15.0], dtype = float32)#TODO fill in the output
    
    >>> device.read("WORD_CLK_MUX", 1, 2 )
    array([13.0], dtype = float32)

    >>> device.read("WORD_CLK_MUX", 0, 2 )
    array([13.0, 12.0], dtype = float32)
        
    >>> device.read("WORD_CLK_MUX", numberOfElementsToRead=1, offsetFromRegisterBaseAddress=2 )
    array([13.0], dtype = float32)
    
    >>> device.read("WORD_CLK_MUX", offsetFromRegisterBaseAddress=2 )
    array([13.0, 12.0], dtype = float32)
    
    See Also
    --------
    Device.readRaw : Read in 'raw' bit values from a device register 

    """
    return numpy.array([0], dtype = numpy.float32)
  
  def write(self, registerName, dataToWrite, offsetFromRegisterBaseAddress=0):
    """ Sets data into a desired register
    
    This method writes values into a register on the board. The method
    internally uses a fixed point converter that is aware of the register width
    on the device and its fractional representation. This Fixed point converter
    converts the double input into corresponding Fixed Point representaions that
    fit into the decive register.
    
    Parameters
    ----------
    registerName : str
      Mapped name of the register to which write access is sought
      
    dataToWrite : numpy.array(dtype = numpy.float32)  
    A numpy array holding the  the values to be written in to the register. The
    array is expected to hold numpy.float32 values. Each value in this array
    represents an induvidual element of the register
       
    offsetFromRegisterBaseAddress : int, optional
      This is an offset from the start of the specified register's base address.
      An offset of 1 represents 32 bits. When an offset is provided as a
      parameter, the method starts to write elements from this address offset
      within the register.
    
    Returns
    -------
    None
    
    Raises
    ------
    datatype #TODO
    
    Examples
    --------
    In the examples, register "WORD_CLK_MUX" is 4 elements long.
    >>> device = Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
    
    >>> dataToWrite = numpy.array([15.0, 14.0, 13.0, 12.0], dtype = float32)
    >>> device.write("WORD_CLK_MUX", dataToWrite)
    
    >>> dataToWrite = numpy.array([13.0, 12.0], dtype = float32)
    >>> device.write("WORD_CLK_MUX", dataToWrite, 2)
    
    """
    return None
  
  def readRaw(self, registerName, numberOfElementsToRead=1, 
              offsetFromRegisterBaseAddress=0):
    """ Returns the raw values from a device's register
    
    This method returns the raw bit values contained in the queried register.
    The returned values are not Fixed Point converted, but direct binary values
    contained in the register elements.
    
    Parameters
    ----------
    registerName : str
      The name of the device register to read from
      
     numberOfElementsToRead : int, optional
      Optional parameter specifying the number of register elements that should
      be read out.
      
      The method returns all elements in the register if this parameter is
      ommitted or when its value is set as 0.
    
     offsetFromRegisterBaseAddress : int, optional
      This is an offset from the start of the specified register's base address.
      An offset of 1 represents 32 bits. When an offset is provided as a
      parameter, the method reads out elements from this point in memory
      (/address offset) onwards.
    
    Returns
    -------
    numpy.array, dtype == numpy.int32
      The method returns a numpy.int32 array containing the raw bit values of
      the register elements. The length of the array either equals the number of
      elements that make up the register or the number specified through the
      numberOfElementsToRead parameter
    
    Raises
    ------
    
    Examples
    --------
    In the example, register "WORD_CLK_MUX" is 4 elements long.
    >>> device = Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
    
    >>> device.read("WORD_CLK_MUX")
    array([15, 14, 13, 12], dtype = int32)#TODO fill in the output
    
    >>> device.read("WORD_CLK_MUX", 0)
    array([15, 14, 13, 12], dtype = int32)#TODO fill in the output
        
    >>> device.read("WORD_CLK_MUX", 1)
    array([15], dtype = int32)#TODO fill in the output
    
    >>> device.read("WORD_CLK_MUX", 1, 2 )
    array([13], dtype = int32)

    >>> device.read("WORD_CLK_MUX", 0, 2 )
    array([13, 12], dtype = int32)
        
    >>> device.read("WORD_CLK_MUX", numberOfElementsToRead=1, offsetFromRegisterBaseAddress=2 )
    array([13], dtype = int32)
    
    >>> device.read("WORD_CLK_MUX", offsetFromRegisterBaseAddress=2 )
    array([13, 12], dtype = int32)
    
    See Also
    --------
    Device.read : Read in Fixed Point converted bit values from a device
    register

    """
    return numpy.array([0], dtype = numpy.int32)
  
  def writeRaw(self, registerName, dataToWrite,
      offsetFromRegisterBaseAddress=0):
    """ Write raw bit values into the register
    
    Provides a way to put in a desired bit value into individual register
    elements. 
    
    Parameters
    ----------
    registerName : str
      The name of the desired register to write into.
      
    dataToWrite : numpy.array, dtype == numpy.int32
     The array holding the bit values to be written into the register. The numpy
     array is expected to contain numpy.int32 values
     
    offsetFromRegisterBaseAddress : int, optional
      This is an offset from the start of the specified register's base address.
      An offset of 1 represents 32 bits. When an offset is provided as a
      parameter, the method starts to write elements from this address offset
      within the register.  
    
    Returns
    -------
    None
    
    Raises
    ------
    datatype #TODO
    
    Examples
    --------
    In the examples, register "WORD_CLK_MUX" is 4 elements long.
    >>> device = Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
    
    >>> dataToWrite = numpy.array([15, 14, 13, 12], dtype = float32)
    >>> device.write("WORD_CLK_MUX", dataToWrite)
    
    >>> dataToWrite = numpy.array([13, 12], dtype = float32)
    >>> device.write("WORD_CLK_MUX", dataToWrite, 2)
        
    
    
    """
    return None
  

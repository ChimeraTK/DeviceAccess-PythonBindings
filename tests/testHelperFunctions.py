#!/usr/bin/env ${python_interpreter}

import  unittest
import numpy
import sys
import os

sys.path.insert(0,os.path.abspath(os.curdir))
import testmodule

class TestHelpers(unittest.TestCase):
    
  def testExtractDataType(self):
    a = numpy.empty([1], dtype = numpy.int32)
    self.assertEqual(testmodule.extractDataType(a), 
                     testmodule.numpyDataTypes.INT32)
    
    a = numpy.empty([1, 2], dtype = numpy.int32)
    self.assertEqual(testmodule.extractDataType(a), 
                     testmodule.numpyDataTypes.INT32)

    a = numpy.empty([1, 2], dtype = numpy.int64)
    self.assertEqual(testmodule.extractDataType(a), 
                     testmodule.numpyDataTypes.INT64)
    
    a = numpy.empty([1, 2], dtype = numpy.float32)
    self.assertEqual(testmodule.extractDataType(a), 
                     testmodule.numpyDataTypes.FLOAT32)
    
    a = numpy.empty([1, 2], dtype = numpy.float64)
    self.assertEqual(testmodule.extractDataType(a), 
                     testmodule.numpyDataTypes.FLOAT64)
    
    a = numpy.empty([1, 2], dtype = numpy.bool)
    self.assertEqual(testmodule.extractDataType(a), 
                     testmodule.numpyDataTypes.USUPPORTED_TYPE)
    
if __name__ == '__main__':
    unittest.main()     
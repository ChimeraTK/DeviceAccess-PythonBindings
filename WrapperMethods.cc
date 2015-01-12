#include "WrapperMethods.h"
#include <vector>


int32_t readReg(mtca4u::devPCIE& self, uint32_t registerOffset, uint8_t bar) {
  int32_t registerContent;
  self.readReg(registerOffset, &registerContent, bar);
  return registerContent;
}

boost::python::list readDMA (mtca4u::devPCIE& self, uint32_t regOffset, size_t size, uint8_t bar) {
  // Error Check if multiple of 4 (or sizeof(WORD_SIZE_IN_BYTES))
  //
  std::vector<int32_t> intBuffer((size)/4); // TODO: Fix this with sizeof?
  boost::python::list readInValues;
  self.readDMA(regOffset, &intBuffer[0], size, bar);

  std::vector<int32_t>::iterator bufferIterator;
  for(bufferIterator = intBuffer.begin(); bufferIterator != intBuffer.end(); bufferIterator++){
    readInValues.append<int32_t>(*bufferIterator);
  }

  return readInValues;
}

boost::python::list readArea(mtca4u::devPCIE& self, int32_t regOffset, size_t size, uint8_t bar) {
  // TODO: Clean this and above method ...

  std::vector<int32_t> intBuffer((size)/4); // TODO: Fix this with sizeof?
  boost::python::list readInValues;
  self.readArea(regOffset, &intBuffer[0], size, bar);

  std::vector<int32_t>::iterator bufferIterator;
  for(bufferIterator = intBuffer.begin(); bufferIterator != intBuffer.end(); bufferIterator++){
    readInValues.append<int32_t>(*bufferIterator);
  }

  return readInValues;

}


void writeArea (mtca4u::devPCIE& self, uint32_t regOffset, boost::python::list data, size_t size,
	   uint8_t bar) {
  int lengthOfList =  boost::python::len(data);
  std::vector<int32_t> intBuffer(lengthOfList); // TODO: Fix this with sizeof?
  std::vector<int32_t>::iterator bufferIterator = intBuffer.begin();
  for(int i =0; i < lengthOfList; i++){
    intBuffer[i] = boost::python::extract<int32_t>(data[i]); // TODO: clean this up
  }
  self.writeArea(regOffset, &intBuffer[0], size, bar);
}



std::string readDeviceInfo(mtca4u::devPCIE & self){
  std::string a;
  self.readDeviceInfo(&a);
  return a;
}


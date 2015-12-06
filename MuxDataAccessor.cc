#include "PythonModuleMethods.h"
#include "PythonExceptions.h"

namespace mtca4upy {
namespace MuxDataAccessor {

  void readInDataFromCard(mtca4u::MultiplexedDataAccessor<float>& self) {
    self.read();
  }

  size_t getSequenceCount(mtca4u::MultiplexedDataAccessor<float>& self) {
    return (self.getNumberOfDataSequences());
  }

  size_t getBlockCount(mtca4u::MultiplexedDataAccessor<float>& self) {
    // FIXME: Make sure prepareAccessor was called. check and throw an exception
    // if
    // it was not
    return (self[0].size());
  }

  void copyReadInData(mtca4u::MultiplexedDataAccessor<float>& self,
                      bp::numeric::array& numpyArray) {
    // FIXME: Make sure prepareAccessor was called. check and throw an exception
    // if
    // it was not
    float* data = reinterpret_cast<float*>(extractDataPointer(numpyArray));

    size_t numSequences = self.getNumberOfDataSequences();
    size_t numBlocks = self[0].size();

    size_t pyArrayRowStride = numSequences;
    for (size_t pyArrayCol = 0; pyArrayCol < numSequences; ++pyArrayCol) {
      for (size_t pyArrrayRow = 0; pyArrrayRow < numBlocks; ++pyArrrayRow) {
        // pyArrayCol corresponds to the sequence numbers and pyArrrayRow to
        // each
        // element of the sequence
        data[(pyArrayRowStride * pyArrrayRow) + pyArrayCol] =
            self[pyArrayCol][pyArrrayRow];
      }
    }
  }
}
}

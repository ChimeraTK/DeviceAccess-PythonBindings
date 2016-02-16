#include "PythonModuleMethods.h"
#include "PythonExceptions.h"

namespace mtca4upy {
namespace MuxDataAccessor {

  size_t getSequenceCount(mtca4u::TwoDRegisterAccessor<float>& self) {
    return (self.getNumberOfDataSequences());
  }

  size_t getBlockCount(mtca4u::TwoDRegisterAccessor<float>& self) {
    return (self[0].size());
  }

  void copyReadInData(mtca4u::TwoDRegisterAccessor<float>& self,
                      bp::numeric::array& numpyArray) {


    float* data = reinterpret_cast<float*>(extractDataPointer(numpyArray));

    size_t numSequences = self.getNumberOfDataSequences();
    size_t numBlocks = self[0].size();
    // Read in data to copy to the numpy array.
    self.read();

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

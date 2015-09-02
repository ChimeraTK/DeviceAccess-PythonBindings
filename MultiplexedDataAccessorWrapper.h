#ifndef SOURCE_DIRECTORY__MULTIPLEXEDDATAACCESSORWRAPPER_H_
#define SOURCE_DIRECTORY__MULTIPLEXEDDATAACCESSORWRAPPER_H_

#include <MtcaMappedDevice/MultiplexedDataAccessor.h>
#include <boost/python.hpp>

namespace mtca4upy {

	// http://dbp-consulting.com/tutorials/SuppressingGCCWarnings.html
	// should temporarily disable the -Weffc++ flag
	// needed because boost::python::wrapper<mtca4upy::PythonDevice> throws a
	// warning
	// for not having a virtual destructor with -Weffc++
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Weffc++"
class MultiplexedDataAccessorWrapper
    : public mtca4u::MultiplexedDataAccessor<float>,
      public boost::python::wrapper<mtca4u::MultiplexedDataAccessor<float> > {
#pragma GCC diagnostic pop
public:
  MultiplexedDataAccessorWrapper(
      boost::shared_ptr<mtca4u::devBase> const& ioDevice,
      std::vector<mtca4u::FixedPointConverter> const& converters);

  void read();
  void write();
  size_t getNumberOfDataSequences();

  virtual ~MultiplexedDataAccessorWrapper();
};

} // namespace mtca4upy

#endif // SOURCE_DIRECTORY__MULTIPLEXEDDATAACCESSORWRAPPER_H_

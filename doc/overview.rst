Overview
========

What is ChimeraTK DeviceAccess?
-------------------------------

ChimeraTK DeviceAccess is a library designed for reading and writing data from register-based devices.
It provides a unified interface that abstracts the underlying hardware communication protocols,
allowing you to focus on your application logic rather than low-level hardware details.

The Python bindings bring this powerful library to Python developers, enabling seamless integration
with Python-based control systems, data acquisition applications, and scientific computing workflows.


Key Features
~~~~~~~~~~~~

* **Named Register Access**: Access registers by meaningful names rather than numerical addresses
* **Automatic Type Conversion**: Seamless conversion between hardware data types and Python types
* **Flexible Accessor Types**: Support for scalar values, arrays, and complex data structures
* **Synchronized Access**: Transfer groups for atomic read/write operations across multiple registers
* **Data Consistency**: Data consistency groups ensure coherent snapshots of multiple registers
* **Multiple Backends**: Support for various communication protocols through backend plugins
* **Synchronous Operations**: All I/O operations are blocking and straightforward

.. note::

   The library is actively maintained and widely used in accelerator control systems and scientific instruments.


How It Works
------------

The basic workflow is:

1. **Open a device** using a device map file that describes your hardware
2. **Get accessors** for the registers you want to work with
3. **Read/Write data** using the accessor interface
4. **Handle errors** with meaningful exceptions

.. code-block:: python

   # Simple workflow example
   device = deviceaccess.Device("MY_DEVICE")

   # Get an accessor for a register
   temperature = device.getScalarRegisterAccessor("TEMPERATURE_SENSOR")

   # Read data from hardware
   temperature.read()

   # Access the data
   current_temp = float(temperature)

   # Write new data
   device.getScalarRegisterAccessor("SETPOINT").write(42.0)


Common Use Cases
----------------

* **Experimental Control Systems**: Control particle accelerators, beamlines, and experiments
* **Sensor Data Acquisition**: Read sensor values and log measurement data
* **Hardware Testing**: Automated testing of hardware devices and components
* **Scientific Instrumentation**: Integration with measurement and analysis frameworks
* **Real-time Data Processing**: Access hardware data for real-time processing pipelines


Why Python Bindings?
--------------------

Python's rich ecosystem and ease of use make it ideal for:

* Rapid prototyping and experimentation
* Integration with scientific computing stacks (NumPy, SciPy, Matplotlib)
* Building automation and monitoring scripts
* Web-based interfaces and dashboards
* Machine learning and data analysis workflows

The C++ backend ensures high performance while Python's flexibility enables rapid development.


Getting Help
~~~~~~~~~~~~

* Check the :doc:`getting_started` guide for basic usage
* Browse the :doc:`examples` for common patterns
* See :doc:`faq` for frequently asked questions
* Consult :doc:`troubleshooting` for common issues
* Review the :doc:`api_reference` for detailed API documentation

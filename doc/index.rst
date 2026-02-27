ChimeraTK DeviceAccess Python Bindings
======================================

.. toctree::
   :hidden:
   :maxdepth: 3

   overview
   getting_started
   user_guide
   examples
   api_reference
   faq
   troubleshooting


Overview
--------

ChimeraTK DeviceAccess Python Bindings provide Pythonic access to the `ChimeraTK DeviceAccess library`__,
a C++ device access library for register-based devices.

.. _ChimeraTK DeviceAccess library: https://chimeratk.github.io/ChimeraTK-DeviceAccess/tag/html/index.html

The bindings enable Python developers to:

* Access hardware registers by name through an intuitive accessor interface
* Read and write device data with automatic type conversion
* Work with scalar, array, and structured data types
* Utilize transfer groups for synchronized access to multiple registers
* Leverage data consistency groups for coherent data reading

All read and write operations are synchronous and blocking until the data transfer is complete, asynchronous operations are available for push-based data updates.


Quick Start
-----------

To get started with the Python bindings, see the :doc:`getting_started` guide for:

* Installation instructions
* Your first device access example
* Basic accessor usage patterns


Tutorials and Examples
----------------------

Learn by example with our comprehensive tutorials:

* :ref:`basic_example_python` - Access a single register value
* :ref:`array_example_python` - Work with array registers
* :ref:`device_map_example_python` - Using device map files
* :ref:`transfer_groups_python` - Synchronized multi-register access
* :ref:`data_consistency_python` - Reading coherent data


User Guide
----------

For more detailed information, refer to the :doc:`user_guide` which covers:

* Understanding accessors
* Data type conversion
* Error handling and exceptions
* Best practices
* Advanced features


API Reference
-------------

Complete API documentation is available in the :doc:`api_reference` section.

The main module is :doc:`deviceaccess` which provides:

* Device class for opening and managing connections
* Various accessor types for different data structures
* Register information retrieval
* Transfer group and data consistency group management


Questions and Troubleshooting
-----------------------------

* See :doc:`faq` for common questions and answers
* Check :doc:`troubleshooting` for solutions to common issues


Indices and Tables
------------------

* :ref:`genindex` - Index of all classes and functions
* :ref:`modindex` - All modules and submodules
* :ref:`search` - Search this documentation

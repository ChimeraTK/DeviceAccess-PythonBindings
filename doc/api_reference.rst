API Reference
=============

Complete API documentation for the ChimeraTK DeviceAccess Python bindings.


Core Classes
~~~~~~~~~~~~

**Device**
  Main class for opening and managing connections to devices.

**ScalarRegisterAccessor**
  Accessor for single-valued registers.

**OneDRegisterAccessor**
  Accessor for array-valued registers.

**TwoDRegisterAccessor**
  Accessor for 2D array registers.


Type Mapping
~~~~~~~~~~~~

Python types are automatically mapped to hardware types, Numpy types are also supported:

* ``int`` ↔ int32
* ``float`` ↔ float32
* ``str`` ↔ String registers
* ``list`` / ``array`` / ``numpy.ndarray`` ↔ Array registers


Main Module: deviceaccess
--------------------------

.. automodule:: deviceaccess
    :members:
    :show-inheritance:


Legacy Module: mtca4u
---------------------

.. note::

   The ``mtca4u`` module is a legacy interface. New code should use the ``deviceaccess`` module instead.

.. automodule:: mtca4u
    :members:
    :undoc-members:
    :show-inheritance:

See Also
--------

* :doc:`user_guide` for usage patterns and best practices
* :doc:`examples` for practical code samples
* :doc:`getting_started` for installation and basic usage
* :doc:`faq` for common questions

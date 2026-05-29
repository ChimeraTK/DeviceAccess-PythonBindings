Examples
========

This page contains practical examples demonstrating common usage patterns with the DeviceAccess Python bindings.
The actual example code, and the `map` and `dmap` files used in this example can be found in the `tests` folder of the source distribution. The content is also listed in the :ref:`Used Map Files section <used_map_files_section>` section below.

.. _basic_example_python:

Basic Scalar Register Access
-----------------------------

Reading and writing a single register value:

.. literalinclude:: ../tests/testDocExamples.py
   :pyobject: TestDocExamples.simpleScalarAccess
   :lines: 2-
   :dedent: 4


.. _array_example_python:

Working with 1D Accessors
-------------------------

Reading and processing array data:

.. literalinclude:: ../tests/testDocExamples.py
   :pyobject: TestDocExamples.simpleOneDAccess
   :lines: 2-
   :dedent: 4


.. _device_map_example_python:

Using Different Device Backends
--------------------------------

The ChimeraTK library supports multiple backends. Configure them in your device map file:

.. code-block:: text

   # Device map example
    AMC_PCIe    (xdma:xdma/slot6?map=device.map)
    SCPI_Dev    (CommandBasedTCP:lab_dev?map=hw_prep.json&port=50000)
    OPCUADev    (opcua:192.168.1.101?port=16664)
    DummyDev    (dummy?map=device.map)

Then use them the same way in your Python code:

.. code-block:: python

   import deviceaccess

   # Open different backend devices with same API
   dummy = deviceaccess.Device("DummyDev")
   pcie = deviceaccess.Device("AMC_PCIe")
   scpi = deviceaccess.Device("SCPI_Dev")
   opcua = deviceaccess.Device("OPCUADev")

   # All use the same accessor interface
   for device in [dummy, pcie, scpi, opcua]:
       value = device.getScalarRegisterAccessor(int, "MEASUREMENT")
       value.read()
       print(f"Value: {float(value)}")

.. _used_map_files_section:

Used Map Files
--------------

.. literalinclude:: ../tests/documentationExamples/someCrate.dmap
   :caption: Example Crate dMap File

.. literalinclude:: ../tests/documentationExamples/someDummyModule.map
   :caption: Example Module Map File


See Also
--------

* :doc:`user_guide` for in-depth explanations
* :doc:`api_reference` for complete API documentation
* :doc:`faq` for common questions

import unittest
import inspect


class TestDocExamples(unittest.TestCase):

    def simpleScalarAccess(self):
        import deviceaccess as da

        da.setDMapFilePath("documentationExamples/someCrate.dmap")
        device = da.Device("someDummyDevice")
        device.open()

        # Read a value
        temperature = device.getScalarRegisterAccessor(float, "SENSORS.TEMPERATURE")
        temperature.read()
        print(f"Current temperature: {float(temperature)} °C")

        # Write a value
        setpoint = device.getScalarRegisterAccessor(float, "SENSORS.SET_POINT")
        setpoint.set(25.0)
        setpoint.write()

        # Verify the write
        setpoint.read()
        print(f"Setpoint is now: {float(setpoint)} °C")

    def simpleOneDAccess(self):
        import deviceaccess as da
        import numpy as np

        da.setDMapFilePath("documentationExamples/someCrate.dmap")
        device = da.Device("someDummyDevice")
        device.open()

        # Get 1D accessor
        waveform = device.getOneDRegisterAccessor(int, "SENSORS.WAVEFORM")
        waveform.read()

        # Access like a list:
        [print(f"Waveform element {i}: {waveform[i]}") for i in range(len(waveform))]

        # Scale all values by a factor of 2
        waveform *= 2

        # Write back the modified waveform
        waveform.write()

        # Read back to verify
        waveform.read()
        print("Modified waveform data:", waveform)  # accessor can be used as a numpy array or python list

        # slicing also works
        print("First 5 elements:", waveform[:3])
        print("Last 2 elements:", waveform[-2:])
        print("Elements 2 to 4:", waveform[1:4])
        print("Every other element:", waveform[::2])

    def testExamples(self):
        # Get all methods defined directly in this class (not inherited)
        test_methods = [
            getattr(self, name) for name in vars(self.__class__)
            if callable(getattr(self.__class__, name)) and name != 'testExamples' and not name.startswith('_')
        ]

        # Call all discovered methods
        for method in test_methods:
            method()

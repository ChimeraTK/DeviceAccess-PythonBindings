import unittest


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

        # Scale all values by a factor of 2
        waveform *= 2

        # Write back the modified waveform
        waveform.write()

        # Read back to verify
        waveform.read()
        print("Modified waveform data:", waveform)  # accessor can be used as a numpy array or python list

    def testExamples(self):
        self.simpleScalarAccess()
        self.simpleOneDAccess()

"""Basler camera adapter matching HIL-SERL's capture interface."""

import numpy as np


class BaslerCapture:
    def __init__(
        self,
        name,
        serial_number=None,
        dim=(2464, 2056),
        fps=10.0,
        exposure=20000.0,
        gain=0,
        gamma=True,
        black_level=0,
        white_balance=(1.0, 1.0, 1.0),
        offset=None,
        timeout_ms=5000,
        max_num_buffer=5,
    ):
        from pypylon import pylon

        if len(dim) != 2 or min(dim) <= 0:
            raise ValueError("dim must contain a positive width and height")
        if fps <= 0 or exposure <= 0 or timeout_ms <= 0 or max_num_buffer <= 0:
            raise ValueError("camera timing and buffer values must be positive")
        if gain < 0 or black_level < 0:
            raise ValueError("gain and black_level must be non-negative")
        if len(white_balance) != 3 or min(white_balance) <= 0:
            raise ValueError("white_balance must contain positive red, green, blue ratios")
        if offset is not None and (len(offset) != 2 or min(offset) < 0):
            raise ValueError("offset must contain non-negative x and y values")

        self.name = name
        self.timeout_ms = int(timeout_ms)
        self.pylon = pylon
        self.camera = None

        factory = pylon.TlFactory.GetInstance()
        devices = list(factory.EnumerateDevices())
        if serial_number is None:
            if len(devices) != 1:
                raise RuntimeError(
                    "serial_number is required unless exactly one Basler camera is connected"
                )
            device = devices[0]
        else:
            serial_number = str(serial_number)
            device = next(
                (d for d in devices if d.GetSerialNumber() == serial_number), None
            )
            if device is None:
                available = ", ".join(d.GetSerialNumber() for d in devices) or "none"
                raise RuntimeError(
                    f"Basler camera {serial_number} not found; available: {available}"
                )

        try:
            self.camera = pylon.InstantCamera(factory.CreateDevice(device))
            self.camera.Open()
            self.camera.OffsetX.Value = self.camera.OffsetX.Min
            self.camera.OffsetY.Value = self.camera.OffsetY.Min
            self._set_integer(self.camera.Width, dim[0])
            self._set_integer(self.camera.Height, dim[1])
            if offset is None:
                self._center(self.camera.OffsetX)
                self._center(self.camera.OffsetY)
            else:
                self._set_integer(self.camera.OffsetX, offset[0])
                self._set_integer(self.camera.OffsetY, offset[1])

            self.camera.AcquisitionFrameRateEnable.Value = True
            frame_rate = self._writable_node(
                "AcquisitionFrameRate", "AcquisitionFrameRateAbs"
            )
            frame_rate.Value = min(float(fps), frame_rate.Max)
            self.camera.ExposureAuto.Value = "Off"
            exposure_time = self._writable_node("ExposureTime", "ExposureTimeAbs")
            exposure_time.Value = min(
                max(float(exposure), exposure_time.Min), exposure_time.Max
            )
            self.camera.GainAuto.Value = "Off"
            self._set_integer(self.camera.GainRaw, gain)
            self.camera.GammaEnable.Value = bool(gamma)
            self._set_integer(self.camera.BlackLevelRaw, black_level)
            self.camera.BalanceWhiteAuto.Value = "Off"
            for channel, ratio in zip(("Red", "Green", "Blue"), white_balance):
                self.camera.BalanceRatioSelector.Value = channel
                balance_ratio = self.camera.BalanceRatioAbs
                balance_ratio.Value = min(
                    max(float(ratio), balance_ratio.Min), balance_ratio.Max
                )
            self.camera.MaxNumBuffer.Value = int(max_num_buffer)

            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment.Value = pylon.OutputBitAlignment_MsbAligned
            # ponytail: cameras free-run independently; use hardware trigger if frame sync matters.
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception:
            self.close()
            raise

    def _writable_node(self, primary, fallback):
        node = getattr(self.camera, primary, None)
        if node is None or not node.IsWritable():
            node = getattr(self.camera, fallback)
        return node

    @staticmethod
    def _set_integer(node, requested):
        minimum, maximum, increment = int(node.Min), int(node.Max), int(node.Inc)
        value = min(max(int(requested), minimum), maximum)
        node.Value = minimum + (value - minimum) // increment * increment

    @staticmethod
    def _center(node):
        minimum, maximum, increment = int(node.Min), int(node.Max), int(node.Inc)
        node.Value = minimum + (maximum - minimum) // (2 * increment) * increment

    @staticmethod
    def get_device_serial_numbers():
        from pypylon import pylon

        return [
            device.GetSerialNumber()
            for device in pylon.TlFactory.GetInstance().EnumerateDevices()
        ]

    def read(self):
        if self.camera is None or not self.camera.IsGrabbing():
            return False, None
        with self.camera.RetrieveResult(
            self.timeout_ms, self.pylon.TimeoutHandling_ThrowException
        ) as result:
            if not result.GrabSucceeded():
                raise RuntimeError(
                    f"Basler camera {self.name} failed: {result.GetErrorDescription()}"
                )
            frame = self.converter.Convert(result).GetArray()
            return True, np.array(frame, copy=True)

    def close(self):
        if self.camera is None:
            return
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        if self.camera.IsOpen():
            self.camera.Close()
        self.camera = None

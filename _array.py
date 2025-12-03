import warnings
import numpy as np

try:
    import cupy as cp

    CUDA_AVAILABLE = True
    NUM_AVAILABLE_GPUS = cp.cuda.runtime.getDeviceCount()
except ImportError:
    cp = None
    CUDA_AVAILABLE = False
    NUM_AVAILABLE_GPUS = 0
    warnings.warn("CuPy is not installed")


class Array:
    def __init__(self, data, device=None, dtype=None):
        if device is not None:
            if device == "cpu":
                tgt_device = "cpu"
                tgt_device_idx = None
            elif "cuda" in device:
                if not CUDA_AVAILABLE:
                    raise RuntimeError("CUDA is not available")
                tgt_device, tgt_device_idx = self.__parse_cude_str(device_str=device)
                if tgt_device_idx + 1 > NUM_AVAILABLE_GPUS:
                    raise RuntimeError(f"cuda:{tgt_device_idx} does not exist")
        else:
            if hasattr(data, "device"):
                if isinstance(data.device, str):
                    if "cuda" in data.device:
                        tgt_device, tgt_device_idx = self.__parse_cude_str(
                            device_str=data.device
                        )
                    else:
                        tgt_device, tgt_device_idx = "cpu", None
                elif isinstance(data.device, cp.cuda.device.Device):
                    tgt_device, tgt_device_idx = "cuda", data.device.id
            else:
                tgt_device, tgt_device_idx = "cpu", None
        if dtype is None:
            if hasattr(data, "dtype"):
                current_dtype = str(data.dtype)
                if current_dtype == "float64":
                    dtype = "float32"
                elif current_dtype == "int64":
                    dtype = "int32"
                else:
                    dtype = current_dtype
        else:
            if not isinstance(dtype, str):
                dtype = str(dtype)
        if isinstance(data, Array):
            self._array = data
        else:
            self._array = np.array(data)
        src_device = (
            "cpu"
            if isinstance(self._array, np.ndarray)
            else f"cuda:{self._array.device.id}"
        )

    def __parse_cude_str(self, device_str):
        tgt_device = "cuda"
        tgt_device_idx = int(device_str.split(":")[-1]) if ":" in device_str else 0
        return tgt_device, tgt_device_idx

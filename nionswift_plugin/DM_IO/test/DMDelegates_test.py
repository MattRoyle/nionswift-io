from __future__ import annotations

import array
import datetime
import io
import itertools
import logging
import pkgutil
import os
import unittest
import shutil
import types
import typing

import h5py
import numpy
import numpy.typing

from nion.data import DataAndMetadata
from nion.data import Calibration
import nionswift_plugin.DM_IO.DMDelegates
import nionswift_plugin.DM_IO.DM5Utils
from nionswift_plugin.DM_IO import DM5IODelegate


def is_equal(r: typing.Any, data: typing.Any) -> bool:
    if isinstance(r, (list, tuple)):
        return len(r) == len(data) and all(is_equal(x, y) for x, y in zip(r, data))
    elif isinstance(r, dict):
        return r.keys() == data.keys() and all(is_equal(r[k], data[k]) for k in r)
    else:
        return bool(r == data)

class TestDM5ImportExportClass(unittest.TestCase, DM5IODelegate):

    def test_data_write_read_round_trip(self) -> None:
        def db_make_directory_if_needed(directory_path: str) -> None:
            if os.path.exists(directory_path):
                if not os.path.isdir(directory_path):
                    raise OSError("Path is not a directory:", directory_path)
            else:
                os.makedirs(directory_path)

        class numpy_array_type:
            def __init__(self, shape: tuple[int, ...], dtype: numpy.typing.DTypeLike) -> None:
                self.data = numpy.ones(shape, dtype)

            def __enter__(self) -> numpy_array_type:
                return self

            def __exit__(self, exception_type: typing.Optional[typing.Type[BaseException]],
                         value: typing.Optional[BaseException], traceback: typing.Optional[types.TracebackType]) -> \
            typing.Optional[bool]:
                return None

        class h5py_array_type:
            def __init__(self, shape: tuple[int, ...], dtype: numpy.typing.DTypeLike) -> None:
                current_working_directory = os.getcwd()
                self.__workspace_dir = os.path.join(current_working_directory, "__Test")
                db_make_directory_if_needed(self.__workspace_dir)
                self.f = h5py.File(os.path.join(self.__workspace_dir, "file.h5"), "a")
                self.data = self.f.create_dataset("data", data=numpy.ones(shape, dtype))

            def __enter__(self) -> h5py_array_type:
                return self

            def __exit__(self, exception_type: typing.Optional[typing.Type[BaseException]],
                         value: typing.Optional[BaseException], traceback: typing.Optional[types.TracebackType]) -> \
            typing.Optional[bool]:
                self.f.close()
                shutil.rmtree(self.__workspace_dir)
                return None

        array_types = numpy_array_type, h5py_array_type
        dtypes = (numpy.float32, numpy.float64, numpy.complex64, numpy.complex128, numpy.int16, numpy.uint16,
                  numpy.int32, numpy.uint32)
        shape_data_descriptors = (
            ((6,), DataAndMetadata.DataDescriptor(False, 0, 1)),  # spectrum
            ((6, 4), DataAndMetadata.DataDescriptor(False, 1, 1)),  # 1d collection of spectra
            ((6, 8, 10), DataAndMetadata.DataDescriptor(False, 2, 1)),  # 2d collection of spectra
            ((6, 4), DataAndMetadata.DataDescriptor(True, 0, 1)),  # sequence of spectra
            ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2)),  # image
            ((6, 4, 2), DataAndMetadata.DataDescriptor(False, 1, 2)),  # 1d collection of images
            ((6, 5, 4, 2), DataAndMetadata.DataDescriptor(False, 2, 2)),  # 2d collection of images. not possible?
            ((6, 8, 10), DataAndMetadata.DataDescriptor(True, 0, 2)),  # sequence of images
        )
        for version in (3, 4):
            for array_type in array_types:
                for dtype in dtypes:
                    for shape, data_descriptor_in in shape_data_descriptors:
                        for signal_type in (
                        (None, "eels") if data_descriptor_in.datum_dimension_count == 1 else (None,)):
                            with array_type(shape, dtype) as a:
                                data_in = a.data
                                dimensional_calibrations_in = list()
                                for index, dimension in enumerate(shape):
                                    dimensional_calibrations_in.append(
                                        Calibration.Calibration(1.0 + 0.1 * index, 2.0 + 0.2 * index,
                                                                "µ" + "n" * index))
                                intensity_calibration_in = Calibration.Calibration(4, 5, "six")
                                metadata_in = dict[typing.Any, typing.Any]()
                                if signal_type:
                                    metadata_in.setdefault("hardware_source", dict())["signal_type"] = signal_type
                                xdata_in = DataAndMetadata.new_data_and_metadata(data_in,
                                                                                 data_descriptor=data_descriptor_in,
                                                                                 dimensional_calibrations=dimensional_calibrations_in,
                                                                                 intensity_calibration=intensity_calibration_in,
                                                                                 metadata=metadata_in)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
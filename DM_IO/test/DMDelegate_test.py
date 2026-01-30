from __future__ import annotations

import abc
import datetime
import io
import itertools
import pkgutil
import os
import shutil
import types
import typing

import h5py
import numpy
import numpy.typing

from nion.data import DataAndMetadata
from nion.data import Calibration
from DM_IO import DMDelegate


class TestDMImportExportBase(abc.ABC):
    @property
    @abc.abstractmethod
    def dm_delegate(self) -> DMDelegate.DMIODelegate:
        ...

    @property
    @abc.abstractmethod
    def versions(self) -> list[int]:
        ...

    @abc.abstractmethod
    def assert_almost_equal(self, actual: typing.Any, desired: typing.Any, decimal: int = 7, err_msg: str | None = None) -> None:
        ...

    @abc.abstractmethod
    def assert_equal(self, actual: typing.Any, desired: typing.Any, err_msg: str | None = None) -> None:
        ...

    @abc.abstractmethod
    def assert_true(self, expr: typing.Any, err_msg: str | None = None) -> None:
        ...

    def calibrations_equal(self, actual_dimension: Calibration.Calibration, desired_dimension: Calibration.Calibration) -> None:
        self.assert_almost_equal(actual_dimension.offset, desired_dimension.offset, 6)
        self.assert_almost_equal(actual_dimension.scale, desired_dimension.scale, 6)
        self.assert_equal(actual_dimension.units, desired_dimension.units)

    def dimension_calibrations_equal(self, actual: typing.Sequence[Calibration.Calibration],
                                     desired: typing.Sequence[Calibration.Calibration]) -> None:
        for actual_dimension, desired_dimension in zip(actual, desired):
            self.calibrations_equal(actual_dimension, desired_dimension)

    def metadata_equal(self, metadata_r: typing.Mapping[str, typing.Any], metadata_l: typing.Mapping[str, typing.Any]) -> None:
        if metadata_r.get("dm_metadata"):
            metadata_r = dict(metadata_r)
            metadata_r.pop('dm_metadata')
        if metadata_l.get("dm_metadata"):
            metadata_l = dict(metadata_l)
            metadata_l.pop('dm_metadata')
        self.assert_equal(metadata_r, metadata_l)

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
            def __init__(self, arr_shape: tuple[int, ...], arr_dtype: numpy.typing.DTypeLike) -> None:
                current_working_directory = os.getcwd()
                self.__workspace_dir = os.path.join(current_working_directory, "__Test")
                db_make_directory_if_needed(self.__workspace_dir)
                self.s = io.BytesIO()
                self.f = h5py.File(self.s, "a")
                if self.f.get('data') is not None:
                    del self.f["data"]
                self.data = self.f.require_dataset("data", data=numpy.ones(arr_shape, arr_dtype), shape=arr_shape,
                                                   dtype=arr_dtype)

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
        for version in self.versions:
            for array_type in array_types:
                for dtype in dtypes:
                    for shape, data_descriptor_in in shape_data_descriptors:
                        for signal_type in (
                                (None, "eels") if data_descriptor_in.datum_dimension_count == 1 else (None,)):
                            with array_type(shape, dtype) as a:
                                with io.BytesIO() as s:
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
                                    self.dm_delegate.save_image(xdata_in, s, version)
                                    s.seek(0)
                                    xdata = self.dm_delegate.load_image(s)
                                    self.assert_true(numpy.array_equal(data_in, xdata.data))
                                    self.assert_equal(data_descriptor_in, xdata.data_descriptor)
                                    self.dimension_calibrations_equal(dimensional_calibrations_in,
                                                                      xdata.dimensional_calibrations)

    def test_rgb_data_write_read_round_trip(self) -> None:

        for version in self.versions:
            s = io.BytesIO()
            data_in = (numpy.random.randn(6, 4, 3) * 255).astype(numpy.uint8)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
            intensity_calibration_in = Calibration.Calibration(4, 5, "six")
            metadata_in: dict[str, typing.Any] = {"abc": None, "": "", "one": [], "two": {}, "three": [1, None, 2]}
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in,
                                                             dimensional_calibrations=dimensional_calibrations_in,
                                                             intensity_calibration=intensity_calibration_in,
                                                             metadata=metadata_in)
            self.dm_delegate.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = self.dm_delegate.load_image(s)
            self.assert_true(numpy.array_equal(data_in, xdata.data))
            self.assert_equal(data_descriptor_in, xdata.data_descriptor)

    def test_calibrations_write_read_round_trip(self) -> None:

        for version in self.versions:
            s = io.BytesIO()
            data_in = numpy.ones((6, 4), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = (Calibration.Calibration(1.1, 2.1, "nm"),
                                           Calibration.Calibration(2, 3, u"µm"))
            intensity_calibration_in = Calibration.Calibration(4.4, 5.5, "six")
            metadata_in = dict[str, typing.Any]()
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in,
                                                             dimensional_calibrations=dimensional_calibrations_in,
                                                             intensity_calibration=intensity_calibration_in,
                                                             metadata=metadata_in)
            self.dm_delegate.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = self.dm_delegate.load_image(s)
            self.dimension_calibrations_equal(dimensional_calibrations_in, xdata.dimensional_calibrations)
            self.calibrations_equal(intensity_calibration_in, xdata.intensity_calibration)

    def test_data_timestamp_write_read_round_trip(self) -> None:

        for version in self.versions:
            s = io.BytesIO()
            data_in = numpy.ones((6, 4), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = [Calibration.Calibration(1.1, 2.1, "nm"),
                                           Calibration.Calibration(2, 3, u"µm")]
            intensity_calibration_in = Calibration.Calibration(4.4, 5.5, "six")
            metadata_in = dict[str, typing.Any]()
            timestamp_in = datetime.datetime(2013, 11, 18, 14, 5, 4, 0)
            timezone_in = "America/Los_Angeles"
            timezone_offset_in = "-0700"
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in,
                                                             dimensional_calibrations=dimensional_calibrations_in,
                                                             intensity_calibration=intensity_calibration_in,
                                                             metadata=metadata_in, timestamp=timestamp_in,
                                                             timezone=timezone_in, timezone_offset=timezone_offset_in)
            self.dm_delegate.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = self.dm_delegate.load_image(s)
            self.assert_equal(timestamp_in, xdata.timestamp)
            self.assert_equal(timezone_in, xdata.timezone)
            self.assert_equal(timezone_offset_in, xdata.timezone_offset)

    def test_metadata_write_read_round_trip(self) -> None:

        for version in self.versions:
            s = io.BytesIO()
            data_in = numpy.ones((6, 4), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
            intensity_calibration_in = Calibration.Calibration(4, 5, "six")
            metadata_in = {
                "abc": 1, "def": "abc",
                "efg": {
                    "one": 1, "two": "TWO",
                    "three": [3, 4, 5], "threef": [3.0, 4.0, 5.0],
                    "four": (32, 32), "fourf": (33.0, 34.0),
                    "six": ((1, 2), (3, 4)), "sixf": ((1.0, 2.0), (3.0, 4.0)),
                    "seven": [[1, 2], [3, 4]], "sevenf": [[1.0, 2.0], [3.0, 4.0]],
                    # the following will not work until there is a schema or other type hinting to distinguish
                    # this from the "six" case.
                    # "eight": (1, 2, 3, 4), "eightf": (1.0, 2.0, 3.0, 4.0),
                    # "eight": (1, 2, 3, 4), "eightf": (1.0, 2.0, 3.0, 4.0),
                }
            }
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in,
                                                             dimensional_calibrations=dimensional_calibrations_in,
                                                             intensity_calibration=intensity_calibration_in,
                                                             metadata=metadata_in)
            self.dm_delegate.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = self.dm_delegate.load_image(s)
            self.metadata_equal(metadata_in, xdata.metadata)

    def test_metadata_difficult_types_write_read_round_trip(self) -> None:

        for version in self.versions:
            s = io.BytesIO()
            data_in = numpy.ones((6, 4), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
            intensity_calibration_in = Calibration.Calibration(4, 5, "six")

            metadata_in: dict[str, typing.Any] = {"abc": None, "": "", "one": [], "two": {}, "three": [1, None, 2]}
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in,
                                                             dimensional_calibrations=dimensional_calibrations_in,
                                                             intensity_calibration=intensity_calibration_in,
                                                             metadata=metadata_in)
            self.dm_delegate.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = self.dm_delegate.load_image(s)
            metadata_expected = {"one": [], "two": {}, "three": [1, 2]}
            self.metadata_equal(metadata_expected, xdata.metadata)

    def test_metadata_export_large_integer(self) -> None:

        for version in self.versions:
            s = io.BytesIO()
            data_in = numpy.ones((6, 4), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 2)
            dimensional_calibrations_in = [Calibration.Calibration(1, 2, "nm"), Calibration.Calibration(2, 3, u"µm")]
            intensity_calibration_in = Calibration.Calibration(4, 5, "six")
            metadata_in: dict[str, typing.Any] = {"abc": 999999999999}
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in,
                                                             dimensional_calibrations=dimensional_calibrations_in,
                                                             intensity_calibration=intensity_calibration_in,
                                                             metadata=metadata_in)
            self.dm_delegate.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = self.dm_delegate.load_image(s)
            metadata_expected = {"abc": 999999999999}
            self.metadata_equal(metadata_expected, xdata.metadata)

    def test_signal_type_round_trip(self) -> None:

        for version in self.versions:
            s = io.BytesIO()
            data_in = numpy.ones((12,), numpy.float32)
            data_descriptor_in = DataAndMetadata.DataDescriptor(False, 0, 1)
            dimensional_calibrations_in = [Calibration.Calibration(1, 2, "eV")]
            intensity_calibration_in = Calibration.Calibration(4, 5, "e")
            metadata_in = {"hardware_source": {"signal_type": "EELS"}}
            xdata_in = DataAndMetadata.new_data_and_metadata(data_in, data_descriptor=data_descriptor_in,
                                                             dimensional_calibrations=dimensional_calibrations_in,
                                                             intensity_calibration=intensity_calibration_in,
                                                             metadata=metadata_in)
            self.dm_delegate.save_image(xdata_in, s, version)
            s.seek(0)
            xdata = self.dm_delegate.load_image(s)
            metadata_expected = {'hardware_source': {'signal_type': 'EELS'},
                                 'Meta Data': {'Format': 'Spectrum', 'Signal': 'EELS'}}
            self.metadata_equal(metadata_expected, xdata.metadata)

    def test_reference_images_load_properly(self) -> None:

        shape_data_descriptors = (
            ((3,), DataAndMetadata.DataDescriptor(False, 0, 1)),  # spectrum
            ((3, 2), DataAndMetadata.DataDescriptor(False, 1, 1)),  # 1d collection of spectra
            ((3, 4, 5), DataAndMetadata.DataDescriptor(False, 2, 1)),  # 2d collection of spectra
            ((3, 2), DataAndMetadata.DataDescriptor(True, 0, 1)),  # sequence of spectra
            ((3, 2), DataAndMetadata.DataDescriptor(False, 0, 2)),  # image
            ((4, 3, 2), DataAndMetadata.DataDescriptor(False, 1, 2)),  # 1d collection of images
            ((3, 4, 5), DataAndMetadata.DataDescriptor(True, 0, 2)),  # sequence of images
        )
        for (shape, data_descriptor), version in itertools.product(shape_data_descriptors, self.versions):
            dimensional_calibrations = list()
            for index, dimension in enumerate(shape):
                dimensional_calibrations.append(
                    Calibration.Calibration(1.0 + 0.1 * index, 2.0 + 0.2 * index, "µ" + "n" * index))
            intensity_calibration = Calibration.Calibration(4, 5, "six")
            data = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape)

            name = f"ref_{'T' if data_descriptor.is_sequence else 'F'}_{data_descriptor.collection_dimension_count}_{data_descriptor.datum_dimension_count}.dm{version}"

            try:
                _data = pkgutil.get_data(__name__, f"resources/{name}")
                assert _data is not None
                s = io.BytesIO(_data)
                xdata = self.dm_delegate.load_image(s)
                self.assert_almost_equal(intensity_calibration.scale, xdata.intensity_calibration.scale, 6)
                self.assert_almost_equal(intensity_calibration.offset, xdata.intensity_calibration.offset, 6)
                self.dimension_calibrations_equal(dimensional_calibrations, xdata.dimensional_calibrations)
                self.calibrations_equal(intensity_calibration, xdata.intensity_calibration)
                self.assert_equal(data_descriptor, xdata.data_descriptor)
                self.assert_true(numpy.array_equal(data, xdata.data))
            except Exception as e:
                print(f"{name} {data_descriptor} failed with exception {e}")
                raise

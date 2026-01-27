"""
    Shared code for supported DM imports and exports
"""
# standard libraries
import gettext
import pathlib
import typing
import abc
import datetime

from nion.data import DataAndMetadata
from nion.data import Calibration


_ = gettext.gettext


class DMIODelegate(abc.ABC):

    @property
    @abc.abstractmethod
    def io_handler_id(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def io_handler_name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def io_handler_extensions(self) -> list[str]:
        ...

    def read_data_and_metadata(self, extension: str, file_path: str) -> DataAndMetadata.DataAndMetadata:
        with open(file_path, "rb", buffering=8 * 1024 * 1024) as f:
            return self.load_image(f)

    def can_write_data_and_metadata(self, data_and_metadata: DataAndMetadata.DataAndMetadata, extension: str) -> bool:
        return extension.lower() in self.io_handler_extensions

    def write_data_and_metadata(self, data_and_metadata: DataAndMetadata.DataAndMetadata, file_path_str: str, extension: str) -> None:
        file_path = pathlib.Path(file_path_str)
        data = data_and_metadata.data
        data_descriptor = data_and_metadata.data_descriptor
        dimensional_calibrations = list()
        for dimensional_calibration in data_and_metadata.dimensional_calibrations:
            offset, scale, units = dimensional_calibration.offset, dimensional_calibration.scale, dimensional_calibration.units
            dimensional_calibrations.append(Calibration.Calibration(offset, scale, units))
        intensity_calibration = data_and_metadata.intensity_calibration
        offset, scale, units = intensity_calibration.offset, intensity_calibration.scale, intensity_calibration.units
        intensity_calibration = Calibration.Calibration(offset, scale, units)
        metadata = data_and_metadata.metadata
        timestamp = data_and_metadata.timestamp
        timezone = data_and_metadata.timezone
        timezone_offset = data_and_metadata.timezone_offset
        version = int(file_path.suffix.strip(".dm"))
        with open(file_path, 'wb', buffering=32 * 1024 * 1024) as f:
            xdata = DataAndMetadata.new_data_and_metadata(data,
                                                          data_descriptor=data_descriptor,
                                                          dimensional_calibrations=dimensional_calibrations,
                                                          intensity_calibration=intensity_calibration,
                                                          metadata=metadata,
                                                          timestamp=timestamp,
                                                          timezone=timezone,
                                                          timezone_offset=timezone_offset)
            self.save_image(xdata, f, version)

    @abc.abstractmethod
    def save_image(self, xdata: DataAndMetadata.DataAndMetadata, file: typing.BinaryIO, file_version: int) -> None:
        ...

    @abc.abstractmethod
    def load_image(self, file: typing.BinaryIO) -> DataAndMetadata.DataAndMetadata:
        ...


def get_datetime_from_timestamp_str(timestamp_str: str) -> typing.Optional[datetime.datetime]:
    if len(timestamp_str) in (23, 26):
        return datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
    elif len(timestamp_str) == 19:
        return datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
    return None
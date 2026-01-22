"""
    Support for DM5 File importing and exporting.
    The thumbnail is not preserved when importing files.
"""

# standard libraries
import gettext
import pathlib
import typing
import datetime

# third party libraries
import h5py
import numpy
import pytz
from pytz import UnknownTimeZoneError
from pytz.exceptions import NonExistentTimeError, AmbiguousTimeError

from nion.data import DataAndMetadata
from nion.data import Calibration
from nionswift_plugin.DM_IO import dm3_image_utils, parse_dm3
from nionswift_plugin.DM_IO import DM5Utils
import abc

from nionswift_plugin.DM_IO.DM5Utils import convert_dict_to_group

_ = gettext.gettext


class DMIODelegate(abc.ABC):

    def __init__(self) -> None:
        self.io_handler_id = "dm-io-handler"
        self.io_handler_name = _("DigitalMicrograph Files")

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

    def save_image(self, xdata: DataAndMetadata.DataAndMetadata, file: typing.BinaryIO, file_version: int) -> None:
        ...

    def load_image(self, file: typing.BinaryIO) -> DataAndMetadata.DataAndMetadata:
        ...


class DM34IODelegate(DMIODelegate):
    def __init__(self):
        super().__init__()

    @property
    def io_handler_extensions(self) -> list[str]:
        return ["dm4", "dm3"]

    def load_image(self, file: typing.BinaryIO) -> DataAndMetadata.DataAndMetadata:
        """
        Loads the image from the file-like object or string file.
        If file is a string, the file is opened and then read.
        Returns a numpy ndarray of our best guess for the most important image
        in the file.
        """
        dmtag = parse_dm3.dm_read_header(file)
        dmtag = dm3_image_utils.fix_strings(dmtag)
        # display_keys(dmtag)
        img_index = -1
        image_tags = dmtag['ImageList'][img_index]
        data = dm3_image_utils.imagedatadict_to_ndarray(image_tags['ImageData'])
        calibrations = list[tuple[float, float, str]]()
        calibration_tags = image_tags['ImageData'].get('Calibrations', dict())
        for dimension in calibration_tags.get('Dimension', list()):
            origin, scale, units = dimension.get('Origin', 0.0), dimension.get('Scale', 1.0), dimension.get('Units',
                                                                                                            str())
            calibrations.append((-origin * scale, scale, units))
        calibrations = list(reversed(calibrations))
        if len(data.shape) == 3 and data.dtype != numpy.uint8:
            if image_tags['ImageTags'].get('Meta Data', dict()).get("Format", str()).lower() in ("spectrum",
                                                                                                 "spectrum image"):
                if data.shape[1] == 1:
                    data = numpy.squeeze(data, 1)
                    data = numpy.moveaxis(data, 0, 1)
                    data_descriptor = DataAndMetadata.DataDescriptor(False, 1, 1)
                    calibrations = [calibrations[2], calibrations[0]]
                else:
                    data = numpy.moveaxis(data, 0, 2)
                    data_descriptor = DataAndMetadata.DataDescriptor(False, 2, 1)
                    calibrations = list(calibrations[1:]) + [calibrations[0]]
            else:
                data_descriptor = DataAndMetadata.DataDescriptor(False, 1, 2)
        elif len(data.shape) == 4 and data.dtype != numpy.uint8:
            # data = numpy.moveaxis(data, 0, 2)
            data_descriptor = DataAndMetadata.DataDescriptor(False, 2, 2)
        elif data.dtype == numpy.uint8:
            data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(data.shape[:-1]))
        else:
            data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(data.shape))
        brightness = calibration_tags.get('Brightness', dict())
        origin, scale, units = brightness.get('Origin', 0.0), brightness.get('Scale', 1.0), brightness.get('Units',
                                                                                                           str())
        intensity = -origin * scale, scale, units
        timestamp = None
        timezone = None
        timezone_offset = None
        properties = dict[str, typing.Any]()
        if 'ImageTags' in image_tags:
            voltage = image_tags['ImageTags'].get('ImageScanned', dict()).get('EHT', dict())
            if voltage:
                properties.setdefault("hardware_source", dict())["autostem"] = {"high_tension": float(voltage)}
            dm_metadata_signal = image_tags['ImageTags'].get('Meta Data', dict()).get('Signal')
            if dm_metadata_signal and dm_metadata_signal.lower() == "eels":
                properties.setdefault("hardware_source", dict())["signal_type"] = dm_metadata_signal
            if image_tags['ImageTags'].get('Meta Data', dict()).get("Format", str()).lower() in ("spectrum",
                                                                                                 "spectrum image"):
                data_descriptor.collection_dimension_count += data_descriptor.datum_dimension_count - 1
                data_descriptor.datum_dimension_count = 1
            if image_tags['ImageTags'].get('Meta Data', dict()).get("IsSequence",
                                                                    False) and data_descriptor.collection_dimension_count > 0:
                data_descriptor.is_sequence = True
                data_descriptor.collection_dimension_count -= 1
            timestamp_str = image_tags['ImageTags'].get("Timestamp")
            if timestamp_str:
                timestamp = dm3_image_utils.get_datetime_from_timestamp_str(timestamp_str)
            timezone = image_tags['ImageTags'].get("Timezone")
            timezone_offset = image_tags['ImageTags'].get("TimezoneOffset")
            # to avoid having duplicate copies in Swift, get rid of these tags
            image_tags['ImageTags'].pop("Timestamp", None)
            image_tags['ImageTags'].pop("Timezone", None)
            image_tags['ImageTags'].pop("TimezoneOffset", None)
            # put the image tags into properties
            properties.update(image_tags['ImageTags'])
        dimensional_calibrations = [Calibration.Calibration(c[0], c[1], c[2]) for c in calibrations]
        while len(dimensional_calibrations) < data_descriptor.expected_dimension_count:
            dimensional_calibrations.append(Calibration.Calibration())
        intensity_calibration = Calibration.Calibration(intensity[0], intensity[1], intensity[2])
        return DataAndMetadata.new_data_and_metadata(data,
                                                     data_descriptor=data_descriptor,
                                                     dimensional_calibrations=dimensional_calibrations,
                                                     intensity_calibration=intensity_calibration,
                                                     metadata=properties,
                                                     timestamp=timestamp,
                                                     timezone=timezone,
                                                     timezone_offset=timezone_offset)

    def save_image(self, xdata: DataAndMetadata.DataAndMetadata, file: typing.BinaryIO, file_version: int) -> None:
        """
        Saves the nparray data to the file-like object (or string) file.
        """
        # we need to create a basic DM tree suitable for an image
        # we'll try the minimum: just an data list
        # doesn't work. Do we need a ImageSourceList too?
        # and a DocumentObjectList?

        data = xdata.data
        data_descriptor = xdata.data_descriptor
        dimensional_calibrations = xdata.dimensional_calibrations
        intensity_calibration = xdata.intensity_calibration
        metadata = xdata.metadata
        modified = xdata.timestamp
        timezone = xdata.timezone
        timezone_offset = xdata.timezone_offset
        needs_slice = False
        is_sequence = False
        if len(data.shape) == 3 and data.dtype != numpy.uint8 and data_descriptor.datum_dimension_count == 1:
            data = numpy.moveaxis(data, 2, 0)
            dimensional_calibrations = (dimensional_calibrations[2],) + tuple(dimensional_calibrations[0:2])
        if len(data.shape) == 2 and data.dtype != numpy.uint8 and data_descriptor.datum_dimension_count == 1:
            is_sequence = data_descriptor.is_sequence
            data = numpy.moveaxis(data, 1, 0)
            data = numpy.expand_dims(data, axis=1)
            dimensional_calibrations = (dimensional_calibrations[1], Calibration.Calibration(),
                                        dimensional_calibrations[0])
            data_descriptor = DataAndMetadata.DataDescriptor(False, 2, 1)
            needs_slice = True
        data_dict = dm3_image_utils.ndarray_to_imagedatadict(data)
        ret = dict[str, typing.Any]()
        ret["ImageList"] = [{"ImageData": data_dict}]
        if dimensional_calibrations and len(dimensional_calibrations) == len(data.shape):
            dimension_list = data_dict.setdefault("Calibrations", dict()).setdefault("Dimension", list())
            for dimensional_calibration in reversed(dimensional_calibrations):
                dimension = dict[str, typing.Any]()
                if dimensional_calibration.scale != 0.0:
                    origin = -dimensional_calibration.offset / dimensional_calibration.scale
                else:
                    origin = 0.0
                dimension['Origin'] = origin
                dimension['Scale'] = dimensional_calibration.scale
                dimension['Units'] = dimensional_calibration.units
                dimension_list.append(dimension)
        if intensity_calibration:
            if intensity_calibration.scale != 0.0:
                origin = -intensity_calibration.offset / intensity_calibration.scale
            else:
                origin = 0.0
            brightness = data_dict.setdefault("Calibrations", dict()).setdefault("Brightness", dict())
            brightness['Origin'] = origin
            brightness['Scale'] = intensity_calibration.scale
            brightness['Units'] = intensity_calibration.units
        if modified:
            timezone_str = None
            if timezone_str is None and timezone:
                try:
                    tz = pytz.timezone(timezone)
                    timezone_str = tz.tzname(modified)
                except (AmbiguousTimeError, NonExistentTimeError):
                    pass
            if timezone_str is None and timezone_offset:
                timezone_str = timezone_offset
            timezone_str = " " + timezone_str if timezone_str is not None else ""
            date_str = modified.strftime("%x")
            time_str = modified.strftime("%X") + timezone_str
            ret["DataBar"] = {"Acquisition Date": date_str, "Acquisition Time": time_str}
        # I think ImageSource list creates a mapping between ImageSourceIds and Images
        ret["ImageSourceList"] = [{"ClassName": "ImageSource:Simple", "Id": [0], "ImageRef": 0}]
        # I think this lists the sources for the DocumentObjectlist. The source number is not
        # the indxe in the imagelist but is either the index in the ImageSourceList or the Id
        # from that list. We also need to set the annotation type to identify it as an data
        ret["DocumentObjectList"] = [{"ImageSource": 0, "AnnotationType": 20}]
        # finally some display options
        ret["Image Behavior"] = {"ViewDisplayID": 8}
        dm_metadata = dict(metadata)
        if dm_metadata.get("dm_metadata"):
            dm_metadata.pop("dm_metadata")
        if metadata.get("hardware_source", dict()).get("signal_type", "").lower() == "eels":
            if len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[0] == 1):
                dm_metadata.setdefault("Meta Data", dict())["Format"] = "Spectrum"
                dm_metadata.setdefault("Meta Data", dict())["Signal"] = "EELS"
        elif data_descriptor.collection_dimension_count == 2 and data_descriptor.datum_dimension_count == 1:
            dm_metadata.setdefault("Meta Data", dict())["Format"] = "Spectrum image"
            dm_metadata.setdefault("Meta Data", dict())["Signal"] = "EELS"
            needs_slice = True
        if data_descriptor.datum_dimension_count == 1:
            # 1d data is always marked as spectrum
            dm_metadata.setdefault("Meta Data", dict())[
                "Format"] = "Spectrum image" if data_descriptor.collection_dimension_count == 2 else "Spectrum"
        if (1 if data_descriptor.is_sequence else 0) + data_descriptor.collection_dimension_count == 1 or needs_slice:
            if data_descriptor.is_sequence or is_sequence:
                dm_metadata.setdefault("Meta Data", dict())["IsSequence"] = True
            ret["ImageSourceList"] = [
                {"ClassName": "ImageSource:Summed", "Do Sum": True, "Id": [0], "ImageRef": 0, "LayerEnd": 0,
                 "LayerStart": 0, "Summed Dimension": len(data.shape) - 1}]
            if needs_slice:
                ret["DocumentObjectList"][0]["AnnotationGroupList"] = [
                    {"AnnotationType": 23, "Name": "SICursor", "Rectangle": (0, 0, 1, 1)}]
                ret["DocumentObjectList"][0]["ImageDisplayType"] = 1  # display as an image
        if modified:
            dm_metadata["Timestamp"] = modified.isoformat()
        if timezone:
            dm_metadata["Timezone"] = timezone
        if timezone_offset:
            dm_metadata["TimezoneOffset"] = timezone_offset
        ret["ImageList"][0]["ImageTags"] = dm_metadata
        ret["InImageMode"] = True
        parse_dm3.dm_write_header(file, file_version, ret)


class DM5IODelegate(DMIODelegate):

    def __init__(self) -> None:
        super().__init__()

    @property
    def io_handler_extensions(self) -> list[str]:
        return ["dm5"]

    def load_image(self, file: typing.BinaryIO) -> DataAndMetadata.DataAndMetadata:
        with (h5py.File(file, "r") as file):
            # Find the index in the image list where the image data is stored
            image_source_index = file.get("DocumentObjectList", dict()).get('[0]', dict()).attrs.get("ImageSource")
            image_ref = file.get("ImageSourceList", dict()).get(f"[{image_source_index}]", dict()).attrs.get("ImageRef")
            image_data = file.get("ImageList").get(f"[{image_ref}]").get("ImageData")
            if None in (image_source_index, image_ref, image_data):
                raise IOError(f"ERROR reading {file.filename}: Malformed file. Unable to determine suitable image source.")

            data = image_data.get("Data", None)
            data = data[()]
            calibrations = list[tuple[float, float, str]]()
            for name, dimension in image_data.get('Calibrations', dict()).get('Dimension', dict()).items():
                origin = dimension.attrs.get('Origin', 0.0)
                scale = dimension.attrs.get('Scale', 1.0)
                units = dimension.attrs.get('Units')
                units_str = ""
                if isinstance(units, bytes):
                    units_str = units.decode()
                calibrations.append((-origin * scale, scale, units_str))
            calibrations = list(reversed(calibrations))

            brightness = image_data.get('Calibrations', dict()).get('Brightness', dict())
            if brightness:
                origin = brightness.attrs.get('Origin', 0.0)
                scale = brightness.attrs.get('Scale', 1.0)
                units = brightness.attrs.get('Units')
                units_str = ""
                if isinstance(units, bytes):
                    units_str = units.decode()
                intensity = -origin * scale, scale, units_str
            intensity_calibration = Calibration.Calibration(intensity[0], intensity[1], intensity[2])

            if file.get("ImageList", dict()).get(f"[{image_ref}]", dict()).get('ImageTags') is None:  # Handle no metadata for image
                dimensional_calibrations = [Calibration.Calibration(c[0], c[1], c[2]) for c in calibrations]
                return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)

            unread_dm_metadata_dict = DM5Utils.convert_group_to_dict(file)
            image_tags = unread_dm_metadata_dict.get("ImageList", dict()).get(f"[{image_ref}]", dict()).get('ImageTags', dict())
            meta_data_attrs = image_tags.get('Meta Data', dict()).get('attrs', dict())
            is_spectrum = meta_data_attrs.get('Format', dict()).get('data', '').lower() in ("spectrum", "spectrum image")
            unique_id = file.get("ImageList").get(f"[{image_ref}]").get('UniqueID')
            if unique_id is not None:
                unique_id_dict = DM5Utils.convert_group_to_dict(unique_id)
                unread_dm_metadata_dict['UniqueID'] = unique_id_dict

            # Logic for the data descriptor
            is_sequence = meta_data_attrs.get('IsSequence', False)
            collection_dimension_count = 0
            datum_dimension_count = len(data.shape)
            if data.dtype == numpy.uint8:
                collection_dimension_count, datum_dimension_count = (0, len(data.shape[:-1]))
            else:
                if len(data.shape) == 3:
                    if is_spectrum:
                        if data.shape[1] == 1:
                            collection_dimension_count, datum_dimension_count = (1, 1)
                            data = numpy.squeeze(data, 1)
                            data = numpy.moveaxis(data, 0, 1)
                            calibrations = [calibrations[2], calibrations[0]]
                        else:
                            collection_dimension_count, datum_dimension_count = (2, 1)
                            data = numpy.moveaxis(data, 0, 2)
                            calibrations = list(calibrations[1:]) + [calibrations[0]]
                    else:
                        collection_dimension_count, datum_dimension_count = (1, 2)
                elif len(data.shape) == 4:
                    collection_dimension_count, datum_dimension_count = (2, 2)

            if is_spectrum:
                collection_dimension_count += datum_dimension_count - 1
                datum_dimension_count = 1

            if is_sequence and collection_dimension_count > 0:
                is_sequence = True
                collection_dimension_count -= 1
            data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)

            properties = dict[str, typing.Any]()

            voltage = image_tags.get('Microscope Info', dict()).get('attrs', dict()).get("Voltage", dict()).get('data')
            if voltage is not None:
                properties.setdefault("hardware_source", dict())["autostem"] = {"high_tension": float(voltage)}

            dm_metadata_signal = meta_data_attrs.get('Signal', dict()).get('data', "")
            if dm_metadata_signal.lower() == "eels":
                properties.setdefault("hardware_source", dict())["signal_type"] = dm_metadata_signal

            data_bar = image_tags.get("Databar", dict())
            timestamp = None

            timestamp_str = image_tags.get('attrs', dict()).get("Timestamp", dict()).get("data")
            if timestamp_str:
                timestamp = dm3_image_utils.get_datetime_from_timestamp_str(timestamp_str)
            timezone = image_tags.get('attrs', dict()).get("Timezone", dict()).get("data")
            timezone_offset = image_tags.get('attrs', dict()).get("TimezoneOffset", dict()).get("data")
            if timestamp_str is None or timezone is None or timezone_offset is None:
                filetime = data_bar.get('attrs', dict()).get('Acquisition Time (OS)', dict()).get('data')
                if filetime is not None:
                    timestamp = DM5Utils.get_datetime_from_filetime(filetime)
                timezone = "UTC"
                timezone_offset = "+0000"

            if image_tags and image_tags.get('attrs'):
                if image_tags.get('attrs', dict()).get('Timestamp'):
                    image_tags['attrs'].pop('Timestamp')
                if image_tags.get('attrs', dict()).get('TimezoneOffset'):
                    image_tags['attrs'].pop('TimezoneOffset')
                if image_tags.get('attrs', dict()).get('Timezone'):
                    image_tags['attrs'].pop('Timezone')
                if len(image_tags['attrs']) == 0:
                    image_tags.pop('attrs')

            properties.update(DM5Utils.squash_metadata_dict(image_tags))

            properties["dm_metadata"] = unread_dm_metadata_dict

            dimensional_calibrations = [Calibration.Calibration(c[0], c[1], c[2]) for c in calibrations]

            while len(dimensional_calibrations) < data_descriptor.expected_dimension_count:
                dimensional_calibrations.append(Calibration.Calibration())

            return DataAndMetadata.new_data_and_metadata(data,
                                                         data_descriptor=data_descriptor,
                                                         dimensional_calibrations=dimensional_calibrations,
                                                         intensity_calibration=intensity_calibration,
                                                         metadata=properties,
                                                         timestamp=timestamp,
                                                         timezone=timezone,
                                                         timezone_offset=timezone_offset)

    def save_image(self, data_and_metadata: DataAndMetadata.DataAndMetadata, file: typing.BinaryIO, file_version: int) -> None:
        data = data_and_metadata.data
        dimensional_calibrations = data_and_metadata.dimensional_calibrations
        intensity_calibration = data_and_metadata.intensity_calibration
        metadata = dict(data_and_metadata.metadata)
        modified = data_and_metadata.timestamp
        timezone = data_and_metadata.timezone
        timezone_offset = data_and_metadata.timezone_offset

        data_descriptor = data_and_metadata.data_descriptor
        is_sequence = data_descriptor.is_sequence
        datum_dimension_count = data_descriptor.datum_dimension_count
        collection_dimension_count = data_descriptor.collection_dimension_count
        needs_slice = False

        if data.dtype != numpy.uint8 and datum_dimension_count == 1:
            if len(data.shape) == 3:
                data = numpy.moveaxis(data, 2, 0)
                dimensional_calibrations = (dimensional_calibrations[2],) + tuple(dimensional_calibrations[0:2])
            if len(data.shape) == 2:
                data = numpy.moveaxis(data, 1, 0)
                data = numpy.expand_dims(data, axis=1)
                dimensional_calibrations = (dimensional_calibrations[1], Calibration.Calibration(), dimensional_calibrations[0])
                collection_dimension_count, datum_dimension_count = (2, 1)
                needs_slice = True

        dm_metadata = metadata.get('dm_metadata', dict())
        if len(dm_metadata) > 0:
            metadata.pop('dm_metadata')
        unique_id = dm_metadata.pop("UniqueID") if dm_metadata.get("UniqueID") is not None else None
        with (h5py.File(file, "w") as f):
            base_group = DM5Utils.convert_dict_to_group(dm_metadata, f)
            image_list = DM5Utils.safe_create_group(base_group, "ImageList")
            source_image = DM5Utils.safe_create_group(image_list, "[1]")  # The image should be in ImageList:[1], 0 is reserved for thumbnails
            image_data = DM5Utils.safe_create_group(source_image, "ImageData")
            image_data.require_dataset("Data", data=data, shape=data.shape, dtype=data.dtype)
            calibrations = DM5Utils.safe_create_group(image_data, "Calibrations")
            if unique_id:
                DM5Utils.convert_dict_to_group(unique_id, source_image)
            # Set up the dimension list with the attributes
            if dimensional_calibrations and len(dimensional_calibrations) == len(data.shape):
                dimension_list = DM5Utils.safe_create_group(calibrations, "Dimension")
                for i, dimensional_calibration in enumerate(reversed(dimensional_calibrations)):
                    origin = 0.0 if dimensional_calibration.scale == 0.0 else -dimensional_calibration.offset / dimensional_calibration.scale
                    dimension = DM5Utils.safe_create_group(dimension_list, f"[{i}]")
                    dimension.attrs.create(name="Origin", data=origin, dtype=numpy.float32)
                    dimension.attrs.create(name="Scale", data=dimensional_calibration.scale, dtype=numpy.float32)  # dm5 stores as float32 however this can introduce floating point issues as python uses 64-bit floats
                    dimension.attrs.create(name="Units", data=numpy.bytes_(dimensional_calibration.units.encode()))

            if intensity_calibration:
                origin = 0.0 if intensity_calibration.scale == 0.0 else -intensity_calibration.offset / intensity_calibration.scale
                brightness = DM5Utils.safe_create_group(calibrations, "Brightness")
                brightness.attrs.create(name="Origin", data=origin, dtype=numpy.float32)
                brightness.attrs.create(name="Scale", data=intensity_calibration.scale, dtype=numpy.float32)
                brightness.attrs.create(name="Units", data=numpy.bytes_(intensity_calibration.units.encode()))

            image_tags = DM5Utils.safe_create_group(source_image, "ImageTags")
            convert_dict_to_group(metadata, image_tags)

            if modified:
                timezone_str = None
                if timezone:
                    try:
                        tz = pytz.timezone(timezone)
                        timezone_str = tz.tzname(modified)
                    except (AmbiguousTimeError, NonExistentTimeError):
                        timezone_str = None

                if timezone_str is None and timezone_offset:
                    timezone_str = timezone_offset

                timezone_str = "" if timezone_str is None else " " + timezone_str
                date_str = modified.strftime("%x")
                time_str = modified.strftime("%X") + timezone_str
                if image_tags.get('Databar') is not None:
                    data_bar = DM5Utils.safe_create_group(image_tags, name="Databar")
                    data_bar.attrs.create(name="Acquisition Date", data=numpy.bytes_(date_str.encode('latin1')))
                    data_bar.attrs.create(name="Acquisition Time", data=numpy.bytes_(time_str.encode('latin1')))

            if modified:
                image_tags.attrs.create(name="Timestamp", data=numpy.bytes_(modified.isoformat().encode('latin1')))
            if timezone:
                image_tags.attrs.create(name="Timezone", data=numpy.bytes_(timezone.encode('latin1')))
            if timezone_offset:
                image_tags.attrs.create(name="TimezoneOffset", data=numpy.bytes_(timezone_offset.encode('latin1')))

            image_source_list = DM5Utils.safe_create_group(base_group, "ImageSourceList")
            image_source = DM5Utils.safe_create_group(image_source_list, "[0]")  # This location is stored in the DocumentObjectList
            image_source.attrs.create(name="ClassName", data=numpy.bytes_("ImageSourceSimple".encode('latin1')))
            image_source.attrs.create(name="ImageRef", data=1, dtype=numpy.uint32)  # The reference in the ImageList
            id_group = DM5Utils.safe_create_group(image_source, name="Id")
            id_group.attrs.create(name="[0]", data=0, dtype=numpy.uint32)

            document_object_list = DM5Utils.safe_create_group(base_group, "DocumentObjectList")
            data_document_object = DM5Utils.safe_create_group(document_object_list, "[0]")
            data_document_object.attrs.create(name="ImageSource", data=0, dtype=numpy.uint64)
            data_document_object.attrs.create(name="AnnotationType", data=20, dtype=numpy.uint32)  # Annotation type 20 is image display

            meta_data_dict: dict[str, typing.Any] = dict()
            meta_data_dict['attrs'] = dict()
            if metadata.get("hardware_source", dict()).get("signal_type", "").lower() == "eels":
                if len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[0] == 1):
                    meta_data_dict['attrs']["Format"] = DM5Utils.data_serialization("Spectrum")
                    meta_data_dict['attrs']["Signal"] = DM5Utils.data_serialization("EELS")
            elif collection_dimension_count == 2 and datum_dimension_count == 1:
                meta_data_dict['attrs']["Format"] = DM5Utils.data_serialization("Spectrum image")
                meta_data_dict['attrs']["Signal"] = DM5Utils.data_serialization("EELS")
                needs_slice = True
            if datum_dimension_count == 1:
                # 1d data is always marked as spectrum
                meta_data_dict['attrs']["Format"] = DM5Utils.data_serialization("Spectrum image" if collection_dimension_count == 2 else "Spectrum")
            if needs_slice or (collection_dimension_count + (1 if is_sequence else 0)) == 1:
                if is_sequence:
                    meta_data_dict['attrs']["IsSequence"] = {"data": True, "dtype": numpy.dtype(numpy.bool_).str}

                image_source.attrs.create(name="ClassName", data=numpy.bytes_("ImageSource:Summed".encode('latin1')))
                image_source.attrs.create(name="Do Sum", data=True)
                image_source.attrs.create(name="LayerEnd", data=0)
                image_source.attrs.create(name="LayerStart", data=0)
                image_source.attrs.create(name="Summed Dimension", data=len(data.shape) - 1)

                if needs_slice:
                    annotation_group_list = DM5Utils.safe_create_group(data_document_object, "AnnotationGroupList")
                    annotation_group = DM5Utils.safe_create_group(annotation_group_list, "[0]")
                    annotation_group.attrs.create(name="AnnotationType", data=23)
                    annotation_group.attrs.create(name="Name", data=numpy.bytes_("SICursor".encode('latin1')))
                    annotation_group.attrs.create(name="Rectangle", data=(0, 0, 1, 1), dtype=[('top', '<f4'), ('left', '<f4'), ('bottom', '<f4'), ('right', '<f4')])
                    data_document_object.attrs.create(name="ImageDisplayType", data=1)

            if len(meta_data_dict['attrs']) != 0 or len(meta_data_dict) > 1:
                meta_data_group = DM5Utils.safe_create_group(image_tags, "Meta Data")
                DM5Utils.convert_dict_to_group(meta_data_dict, meta_data_group)

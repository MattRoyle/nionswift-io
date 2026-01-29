import gettext
import typing

import h5py
import numpy
import pytz

from DM_IO import DM5Utils
from DM_IO import DMDelegate
from nion.data import DataAndMetadata
from nion.data import Calibration
from nionutils.nion.utils import DateTime
_ = gettext.gettext


class DM5IODelegate(DMDelegate.DMIODelegate):

    @property
    def io_handler_id(self) -> str:
        return "dm5-io-handler"

    @property
    def io_handler_name(self) -> str:
        return _("DigitalMicrograph")

    @property
    def io_handler_extensions(self) -> list[str]:
        return ["dm5"]

    def load_image(self, file: typing.BinaryIO) -> DataAndMetadata.DataAndMetadata:
        with h5py.File(file, "r") as file:
            # Find the index in the image list where the image data is stored
            document_object = file.get("DocumentObjectList", dict()).get('[0]', dict())
            if not hasattr(document_object, "attrs"):
                raise IOError(f"ERROR reading {file.filename}: Malformed file. Unable to determine suitable image source as document object had no attributes.")
            image_source_index = document_object.attrs.get("ImageSource")
            image_source = file.get("ImageSourceList", dict()).get(f"[{image_source_index}]", dict())
            if not hasattr(image_source, "attrs"):
                raise IOError(f"ERROR reading {file.filename}: Malformed file. Unable to determine suitable image source as it had no attributes.")
            image_ref = image_source.attrs.get("ImageRef")
            image_data = file.get("ImageList").get(f"[{image_ref}]").get("ImageData")
            if None in (image_source_index, image_ref, image_data):
                raise IOError(f"ERROR reading {file.filename}: Malformed file. Unable to determine suitable image source.")

            data = image_data.get("Data", None)

            if data is None:
                raise IOError(f"ERROR reading {file.filename}: No data found in at image data.")
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

            if file.get("ImageList", dict()).get(f"[{image_ref}]", dict()).get(
                    'ImageTags') is None:  # Handle no metadata for image
                dimensional_calibrations = [Calibration.Calibration(c[0], c[1], c[2]) for c in calibrations]
                return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration,
                                                             dimensional_calibrations=dimensional_calibrations)

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
            data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count,
                                                             datum_dimension_count)

            properties = dict[str, typing.Any]()
            voltage = image_tags.get('Microscope Info', dict()).get('attrs', dict()).get("Voltage", dict()).get("data")
            if voltage is not None:
                properties.setdefault("hardware_source", dict())["autostem"] = {"high_tension": float(voltage)}

            dm_metadata_signal = meta_data_attrs.get('Signal', dict()).get('data', "")
            if dm_metadata_signal.lower() == "eels":
                properties.setdefault("hardware_source", dict())["signal_type"] = dm_metadata_signal

            data_bar = image_tags.get("Databar", dict())
            timestamp = None

            timestamp_str = image_tags.get('attrs', dict()).get("Timestamp", dict()).get("data")
            if timestamp_str:
                timestamp = DMDelegate.get_datetime_from_timestamp_str(timestamp_str)
            timezone = image_tags.get('attrs', dict()).get("Timezone", dict()).get("data")
            timezone_offset = image_tags.get('attrs', dict()).get("TimezoneOffset", dict()).get("data")
            if timestamp_str is None or timezone is None or timezone_offset is None:
                filetime = data_bar.get('attrs', dict()).get('Acquisition Time (OS)', dict()).get('data')
                if filetime is not None:
                    timestamp = DateTime.get_datetime_from_windows_filetime(filetime)
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

    def save_image(self, data_and_metadata: DataAndMetadata.DataAndMetadata, file: typing.BinaryIO, _: int) -> None:
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
                dimensional_calibrations = (dimensional_calibrations[1], Calibration.Calibration(),
                                            dimensional_calibrations[0])
                collection_dimension_count, datum_dimension_count = (2, 1)
                needs_slice = True

        dm_metadata = metadata.get('dm_metadata', dict())
        if len(dm_metadata) > 0:  # in order to preserve as much of a dm5 file structure as possible the importer stores a dict representation in the metadata
            metadata.pop('dm_metadata')  # the dict representation is removed, with the rest of the metadata being used for ImageTags

        with (h5py.File(file, "w") as f):
            base_group = DM5Utils.convert_dict_to_group(dm_metadata, f)
            image_list = DM5Utils.get_or_create_group(base_group, "ImageList")
            source_image = DM5Utils.get_or_create_group(image_list,
                                                        "[1]")  # The image should be in ImageList:[1], 0 is reserved for thumbnails
            image_data = DM5Utils.get_or_create_group(source_image, "ImageData")
            image_data.require_dataset("Data", data=data, shape=data.shape, dtype=data.dtype)
            calibrations = DM5Utils.get_or_create_group(image_data, "Calibrations")

            # Set up the dimension list with the attributes
            if dimensional_calibrations and len(dimensional_calibrations) == len(data.shape):
                dimension_list = DM5Utils.get_or_create_group(calibrations, "Dimension")
                for i, dimensional_calibration in enumerate(reversed(dimensional_calibrations)):
                    origin = 0.0 if dimensional_calibration.scale == 0.0 else -dimensional_calibration.offset / dimensional_calibration.scale
                    dimension = DM5Utils.get_or_create_group(dimension_list, f"[{i}]")
                    DM5Utils.save_attr_to_group(name="Origin", data=origin, dtype=numpy.float32, group=dimension)
                    DM5Utils.save_attr_to_group(name="Scale", data=dimensional_calibration.scale, dtype=numpy.float32,
                                                group=dimension)  # dm5 stores scale as a float32. This can introduce floating point issues as python uses 64-bit floats
                    DM5Utils.save_attr_to_group(name="Units", data=dimensional_calibration.units, group=dimension)

            if intensity_calibration:
                origin = 0.0 if intensity_calibration.scale == 0.0 else -intensity_calibration.offset / intensity_calibration.scale
                brightness = DM5Utils.get_or_create_group(calibrations, "Brightness")
                DM5Utils.save_attr_to_group(name="Origin", data=origin, dtype=numpy.float32, group=brightness)
                DM5Utils.save_attr_to_group(name="Scale", data=intensity_calibration.scale, dtype=numpy.float32,
                                            group=brightness)
                DM5Utils.save_attr_to_group(name="Units", data=intensity_calibration.units, group=brightness)

            image_tags = DM5Utils.get_or_create_group(source_image, "ImageTags")
            DM5Utils.convert_dict_to_group(metadata, image_tags)  # The metadata dictionary is treated as the image tags

            if modified:
                timezone_str = None
                if timezone:
                    try:
                        tz = pytz.timezone(timezone)
                        timezone_str = tz.tzname(modified)
                    except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
                        timezone_str = None

                if timezone_str is None and timezone_offset:
                    timezone_str = timezone_offset

                timezone_str = "" if timezone_str is None else " " + timezone_str
                date_str = modified.strftime("%x")
                time_str = modified.strftime("%X") + timezone_str
                if image_tags.get('Databar') is not None:
                    data_bar = DM5Utils.get_or_create_group(image_tags, name="Databar")
                    DM5Utils.save_attr_to_group(name="Acquisition Date", data=date_str, group=data_bar)
                    DM5Utils.save_attr_to_group(name="Acquisition Time", data=time_str, group=data_bar)

            if modified:
                DM5Utils.save_attr_to_group(name="Timestamp", data=modified.isoformat(), group=image_tags)
            if timezone:
                DM5Utils.save_attr_to_group(name="Timezone", data=timezone, group=image_tags)
            if timezone_offset:
                DM5Utils.save_attr_to_group(name="TimezoneOffset", data=timezone_offset, group=image_tags)

            image_source_list = DM5Utils.get_or_create_group(base_group, "ImageSourceList")
            image_source = DM5Utils.get_or_create_group(image_source_list,
                                                        "[0]")  # This location is stored in the DocumentObjectList
            DM5Utils.save_attr_to_group(name="ClassName", data="ImageSourceSimple", group=image_source)
            DM5Utils.save_attr_to_group(name="ImageRef", data=1, group=image_source,
                                        dtype=numpy.uint32)  # The reference in the ImageList
            id_group = DM5Utils.get_or_create_group(image_source, name="Id")
            DM5Utils.save_attr_to_group(name="[0]", data=0, group=id_group, dtype=numpy.uint32)

            document_object_list = DM5Utils.get_or_create_group(base_group, "DocumentObjectList")
            data_document_object = DM5Utils.get_or_create_group(document_object_list, "[0]")
            DM5Utils.save_attr_to_group(name="ImageSource", data=0, group=data_document_object, dtype=numpy.uint64)
            DM5Utils.save_attr_to_group(name="AnnotationType", data=20, group=data_document_object,
                                        dtype=numpy.uint32)  # Annotation type 20 is image display

            meta_data_group = DM5Utils.get_or_create_group(image_tags, "Meta Data")
            if metadata.get("hardware_source", dict()).get("signal_type", "").lower() == "eels":
                if len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[0] == 1):
                    DM5Utils.save_attr_to_group("Format", data="Spectrum", group=meta_data_group)
                    DM5Utils.save_attr_to_group("Signal", data="EELS", group=meta_data_group)
            elif collection_dimension_count == 2 and datum_dimension_count == 1:
                DM5Utils.save_attr_to_group("Format", data="Spectrum image", group=meta_data_group)
                DM5Utils.save_attr_to_group("Signal", data="EELS", group=meta_data_group)
                needs_slice = True
            if datum_dimension_count == 1:
                # 1d data is always marked as spectrum
                DM5Utils.save_attr_to_group("Format",
                                            data="Spectrum image" if collection_dimension_count == 2 else "Spectrum",
                                            group=meta_data_group)
            if needs_slice or (collection_dimension_count + (1 if is_sequence else 0)) == 1:
                if is_sequence:
                    DM5Utils.save_attr_to_group("IsSequence", data=True, group=meta_data_group)

                DM5Utils.save_attr_to_group(name="ClassName", data="ImageSource:Summed", group=image_source)
                DM5Utils.save_attr_to_group(name="Do Sum", data=True, group=image_source)
                DM5Utils.save_attr_to_group(name="LayerEnd", data=0, group=image_source)
                DM5Utils.save_attr_to_group(name="LayerStart", data=0, group=image_source)
                DM5Utils.save_attr_to_group(name="Summed Dimension", data=len(data.shape) - 1, group=image_source)

                if needs_slice:
                    annotation_group_list = DM5Utils.get_or_create_group(data_document_object, "AnnotationGroupList")
                    annotation_group = DM5Utils.get_or_create_group(annotation_group_list, "[0]")
                    DM5Utils.save_attr_to_group(name="AnnotationType", data=23, group=annotation_group)
                    DM5Utils.save_attr_to_group(name="Name", data="SICursor", group=annotation_group)
                    DM5Utils.save_attr_to_group(name="Rectangle", data=(0, 0, 1, 1), group=annotation_group,
                                                dtype=[('top', '<f4'), ('left', '<f4'), ('bottom', '<f4'),
                                                       ('right', '<f4')])
                    DM5Utils.save_attr_to_group(name="ImageDisplayType", data=1, group=data_document_object)

            if len(meta_data_group.attrs.keys()) == 0:
                del image_tags["Meta Data"]  # If nothing was added to the group remove it

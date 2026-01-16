"""
    Support for DM5 I/O.
"""

# standard libraries
import gettext
import json
import pathlib
import typing

# third party libraries
import h5py
import numpy as np
import pytz
from pytz.exceptions import NonExistentTimeError, AmbiguousTimeError

from nion.data import DataAndMetadata
from nion.data import Calibration
from nionswift_plugin.DM_IO import dm3_image_utils
from nionswift_plugin.DM_IO.DM5Utils import convert_group_to_dict, get_datetime_from_filetime, convert_dict_to_group, \
    safe_create_group

_ = gettext.gettext

class DMIODelegate(object):

    def __init__(self, api: typing.Any) -> None:
        self.__api = api
        self.io_handler_id = "dm-io-handler"
        self.io_handler_name = _("DigitalMicrograph Files")
        self.io_handler_extensions = ["dm4", "dm3"]

    def read_data_and_metadata(self, extension: str, file_path: str) -> DataAndMetadata.DataAndMetadata:
        with open(file_path, "rb", buffering=8 * 1024 * 1024) as f:
            return dm3_image_utils.load_image(f)

    def can_write_data_and_metadata(self, data_and_metadata: DataAndMetadata.DataAndMetadata, extension: str) -> bool:
        return extension.lower() in self.io_handler_extensions

    def write_data_and_metadata(self, data_and_metadata: DataAndMetadata.DataAndMetadata, file_path_str: str, extension: str) -> None:
        file_path = pathlib.Path(file_path_str)
        data = data_and_metadata.data
        data_descriptor = data_and_metadata.data_descriptor
        dimensional_calibrations = list()
        for dimensional_calibration in data_and_metadata.dimensional_calibrations:
            offset, scale, units = dimensional_calibration.offset, dimensional_calibration.scale, dimensional_calibration.units
            dimensional_calibrations.append(self.__api.create_calibration(offset, scale, units))
        intensity_calibration = data_and_metadata.intensity_calibration
        offset, scale, units = intensity_calibration.offset, intensity_calibration.scale, intensity_calibration.units
        intensity_calibration = self.__api.create_calibration(offset, scale, units)
        metadata = data_and_metadata.metadata
        timestamp = data_and_metadata.timestamp
        timezone = data_and_metadata.timezone
        timezone_offset = data_and_metadata.timezone_offset
        with open(file_path, 'wb', buffering=32 * 1024 * 1024) as f:
            xdata = DataAndMetadata.new_data_and_metadata(data,
                                                          data_descriptor=data_descriptor,
                                                          dimensional_calibrations=dimensional_calibrations,
                                                          intensity_calibration=intensity_calibration,
                                                          metadata=metadata,
                                                          timestamp=timestamp,
                                                          timezone=timezone,
                                                          timezone_offset=timezone_offset)
            dm3_image_utils.save_image(xdata, f, 4 if file_path.suffix == ".dm4" else 3)


class DM5IODelegate(DMIODelegate):

    def __init__(self, api: typing.Any) -> None:
        super().__init__(api)
        self.io_handler_extensions = ["dm5"]

    def read_data_and_metadata(self, extension: str, file_path: str) -> DataAndMetadata.DataAndMetadata:
        with (h5py.File(file_path, "r") as file):
            # Find the index in the image list where the image data is stored TODO should the default be dict() or an empty h5py.Group?
            image_source_index = file.get("DocumentObjectList", dict()).get('[0]', dict()).attrs.get("ImageSource")
            image_ref = file.get("ImageSourceList", dict()).get(f"[{image_source_index}]", dict()).attrs.get("ImageRef")
            image_data = file.get("ImageList").get(f"[{image_ref}]").get("ImageData")
            if None in (image_source_index, image_ref, image_data):
                raise IOError(f"ERROR reading {file_path}: Malformed file. Unable to determine suitable image source.")

            data = image_data.get("Data", None)
            #assert(isinstance(data, h5py.Dataset), f"ERROR reading {file_path}: Malformed file. Parsed Image data was not a suitable type.")
            data = data[()]
            calibrations = list[tuple[float, float, str]]()
            for name, dimension in image_data.get('Calibrations', dict()).get('Dimension', dict()).items():
                origin = dimension.attrs.get('Origin', 0.0)
                scale = dimension.attrs.get('Scale', 1.0)git
                units = dimension.attrs.get('Units', "")
                calibrations.append((-origin * scale, scale, units.astype(str)))
            calibrations = list(reversed(calibrations))

            brightness = image_data.get('Calibrations', dict()).get('Brightness', dict())
            if brightness:
                origin = brightness.attrs.get('Origin', 0.0)
                scale = brightness.attrs.get('Scale', 1.0)
                units = brightness.attrs.get('Units', "")
                intensity = -origin * scale, scale, units.astype(str)

            if file.get("ImageList", dict()).get(f"[{image_ref}]", dict()).get('ImageTags') is None: # Handle no metadata for image
                return DataAndMetadata.new_data_and_metadata(data)

            # TODO properties doesn't contain the thumbnail currently, should it?
            unread_dm_metadata_dict = convert_group_to_dict(file)
            image_tags = unread_dm_metadata_dict.get("ImageList", dict()).pop(f"[{image_ref}]", dict()).get('ImageTags', dict())
            meta_data_attrs = image_tags.get('Meta Data', dict()).get('attrs', dict()).get('data', dict())
            is_spectrum = meta_data_attrs.get('Format', "").lower() in ("spectrum", "spectrum image")
            unique_id = file.get("ImageList").get(f"[{image_ref}]").get('UniqueID')
            if unique_id is not None:
                unique_id_dict = convert_group_to_dict(unique_id)
                unread_dm_metadata_dict['UniqueID'] = unique_id_dict

            # Logic for the data descriptor
            is_sequence = meta_data_attrs.get('IsSequence', False)
            collection_dimension_count = 0
            datum_dimension_count = len(data.shape)
            if data.dtype == np.uint8:
                collection_dimension_count, datum_dimension_count = (0, len(data.shape[:-1]))
            else:
                if len(data.shape) == 3:
                    if is_spectrum:
                        if data.shape[1] == 1:
                            collection_dimension_count, datum_dimension_count = (1, 1)
                            data = np.squeeze(data, 1)
                            data = np.moveaxis(data, 0, 1)
                            calibrations = [calibrations[2], calibrations[0]]
                        else:
                            collection_dimension_count, datum_dimension_count = (2, 1)
                            data = np.moveaxis(data, 0, 2)
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

            dm_metadata_signal = meta_data_attrs.get('Signal')
            if dm_metadata_signal and dm_metadata_signal.lower() == "eels":
                properties.setdefault("hardware_source", dict())["signal_type"] = dm_metadata_signal


            data_bar = image_tags.get("Databar", dict())
            timestamp = None
            filetime = data_bar.get('attrs', dict()).get('Acquisition Time (OS)', dict()).get('data')
            if filetime is not None:
                timestamp = get_datetime_from_filetime(int(round(filetime)))
            timezone = "UTC"
            timezone_offset = "0" # Todo verify what the offset looks like?

            properties.update(image_tags)



            properties["dm_metadata"] = unread_dm_metadata_dict

            #json.dump(unread_dm_metadata_dict, open("C:/Temp/unread.json", "w"))
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

    def write_data_and_metadata(self, data_and_metadata: DataAndMetadata.DataAndMetadata, file_path_str: str, extension: str) -> None:
        data = data_and_metadata.data
        dimensional_calibrations = data_and_metadata.dimensional_calibrations
        intensity_calibration = data_and_metadata.intensity_calibration
        metadata = data_and_metadata.metadata
        modified = data_and_metadata.timestamp
        timezone = data_and_metadata.timezone
        timezone_offset = data_and_metadata.timezone_offset

        data_descriptor = data_and_metadata.data_descriptor
        is_sequence = data_descriptor.is_sequence
        datum_dimension_count = data_descriptor.datum_dimension_count
        collection_dimension_count = data_descriptor.collection_dimension_count
        needs_slice = False

        if data.dtype != np.uint8 and datum_dimension_count == 1:
            if len(data.shape) == 2:
                data = np.moveaxis(data, 1, 0)
                data = np.expand_dims(data, axis=1)
                dimensional_calibrations = (dimensional_calibrations[1], Calibration.Calibration(), dimensional_calibrations[0])
                collection_dimension_count, datum_dimension_count = (2, 1)
                needs_slice = True
            elif len(data.shape) == 3:
                data = np.moveaxis(data, 2, 0)
                dimensional_calibrations = (dimensional_calibrations[2],) + tuple(dimensional_calibrations[0:2])

        dm_metadata = metadata.get('dm_metadata', dict())
        unique_id = dm_metadata.pop("UniqueID") if dm_metadata.get("UniqueID") is not None else None
        with (h5py.File(file_path_str, "w") as f):
            base_group = convert_dict_to_group(dm_metadata, f)
            image_list = safe_create_group(base_group, "ImageList")
            source_image = safe_create_group(image_list, "[1]") # The image should be in ImageList:[1], 0 is reserved for thumbnails
            image_data = safe_create_group(source_image, "ImageData")
            image_data.create_dataset("Data", data=data)
            calibrations = safe_create_group(image_data, "Calibrations")
            if unique_id:
                unique_id_group =  convert_dict_to_group(unique_id, source_image)
            # Set up the dimension list with the attributes
            if dimensional_calibrations and len(dimensional_calibrations) == len(data.shape):
                dimension_list = safe_create_group(calibrations, "Dimension")
                for i, dimensional_calibration in enumerate(reversed(dimensional_calibrations)):
                    origin = 0.0 if dimensional_calibration.scale == 0.0 else -dimensional_calibration.scale / dimensional_calibration.scale
                    dimension = safe_create_group(dimension_list, f"[{i}]")
                    dimension.attrs.create(name="Origin", data=origin, dtype=np.float32)
                    dimension.attrs.create(name="Scale", data=dimensional_calibration.scale, dtype=np.float32)
                    dimension.attrs.create(name="Units", data=dimensional_calibration.units)

            if intensity_calibration: # TODO does attributes need the empty label?
                origin = 0.0 if intensity_calibration.scale == 0.0 else -intensity_calibration.scale / intensity_calibration.scale
                brightness = safe_create_group(calibrations, "Brightness")
                brightness.attrs.create(name="Origin", data=origin, dtype=np.float32)
                brightness.attrs.create(name="Scale", data=intensity_calibration.scale, dtype=np.float32)
                brightness.attrs.create(name="Units", data=intensity_calibration.units)

            image_tags = convert_dict_to_group(metadata.get("ImageTags", dict()), source_image)

            if False: # TODO determine where to put the time stamps
                data_bar = safe_create_group(image_tags, name="Databar")
                if modified:
                    timezone_str = None
                    if timezone:
                        try:
                            tz = pytz.timezone(timezone)
                            timezone_str = tz.tzname(modified)
                        except NonExistentTimeError or AmbiguousTimeError:
                            timezone_str = None

                    if timezone_offset is None and timezone_offset:
                        timezone_str = timezone_offset

                    timezone_str = "" if timezone_str is None else " " + timezone_str
                    date_str = modified.strftime("%x")
                    time_str = modified.strftime("%X") + timezone_str
                    data_bar.attrs.create(name="Acquisition Date", data=date_str)
                    data_bar.attrs.create(name="Acquisition Time", data=time_str)

            image_source_list = safe_create_group(base_group, "ImageSourceList")
            image_source = safe_create_group(image_source_list, "[0]") # This location is stored in the DocumentObjectList
            image_source.attrs.create(name="ClassName", data="ImageSourceSimple")
            image_source.attrs.create(name="ImageRef", data=1, dtype=np.uint32) # The reference in the ImageList
            id_group = safe_create_group(image_source, name="Id")
            id_group.attrs.create(name="[0]", data=0, dtype=np.uint32)

            document_object_list = safe_create_group(base_group, "DocumentObjectList")
            data_document_object = safe_create_group(document_object_list, "[0]")
            data_document_object.attrs.create(name="ImageSource", data=0, dtype=np.uint64)
            data_document_object.attrs.create(name="AnnotationType", data=20, dtype=np.uint32)  # Annotation type 20 is image display

            meta_data_dict = dict()
            meta_data_dict['attrs'] = dict()
            if metadata.get("hardware_source", dict()).get("signal_type", "").lower() == "eels":
                if len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[0] == 1):
                    meta_data_dict['attrs']["Format"] = "Spectrum"
                    meta_data_dict['attrs']["Signal"] = "EELS"
            elif collection_dimension_count == 2 and datum_dimension_count == 1:
                meta_data_dict['attrs']["Format"] = "Spectrum image"
                meta_data_dict['attrs']["Signal"] = "EELS"
                needs_slice = True
            if datum_dimension_count == 1:
                # 1d data is always marked as spectrum
                meta_data_dict['attrs']["Format"] = "Spectrum image" if collection_dimension_count == 2 else "Spectrum"
            if needs_slice or (collection_dimension_count + (1 if is_sequence else 0)) == 1:
                if is_sequence:
                    meta_data_dict['attrs']["IsSequence"] = True

                image_source.attrs.create(name="ClassName", data="ImageSource:Summed")
                image_source.attrs.create(name="Do Sum", data=True)
                image_source.attrs.create(name="LayerEnd", data=0)
                image_source.attrs.create(name="LayerStart", data=0)
                image_source.attrs.create(name="Summed Dimension", data=len(data.shape) - 1)

                if needs_slice:
                    annotation_group_list = safe_create_group(data_document_object, "AnnotationGroupList")
                    annotation_group = safe_create_group(annotation_group_list, "[0]")
                    annotation_group.attrs.create(name="AnnotationType", data=23)
                    annotation_group.attrs.create(name="Name", data="SICursor")
                    annotation_group.attrs.create(name="Rectangle", data=(0, 0, 1, 1), dtype=np.void)
                    data_document_object.attrs.create(name="ImageDisplayType", data=1)

            if meta_data_dict:
                meta_data_group = safe_create_group(image_tags, "Meta Data")
                convert_dict_to_group(meta_data_dict, meta_data_group)




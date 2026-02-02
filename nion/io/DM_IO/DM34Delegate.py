import gettext
import typing
import numpy
import pytz

from nion.io.DM_IO import ParseDM34File, DM34ImageUtils, DMDelegate
from nion.data import DataAndMetadata, Calibration

_ = gettext.gettext


class DM34IODelegate(DMDelegate.DMIODelegate):

    @property
    def io_handler_id(self) -> str:
        return "dm34-io-handler"

    @property
    def io_handler_name(self) -> str:
        return _("DigitalMicrograph")

    @property
    def io_handler_extensions(self) -> list[str]:
        return ["dm4", "dm3"]

    def _import_data_and_metadata(self, file: typing.BinaryIO) -> DataAndMetadata.DataAndMetadata:
        """Loads the image from the file-like object or string file.
        If file is a string, the file is opened and then read.
        Returns a numpy ndarray of our best guess for the most important image
        in the file.
        """
        dmtag = ParseDM34File.dm_read_header(file)
        dmtag = DM34ImageUtils.fix_strings(dmtag)
        # display_keys(dmtag)
        img_index = -1
        image_tags = dmtag['ImageList'][img_index]
        data = DM34ImageUtils.imagedatadict_to_ndarray(image_tags['ImageData'])
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
                timestamp = DMDelegate.get_datetime_from_timestamp_str(timestamp_str)
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

    def _export_data_and_metadata(self, xdata: DataAndMetadata.DataAndMetadata, file: typing.BinaryIO, file_version: int) -> None:
        """Saves the nparray data to the file-like object (or string) file.
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
        data_dict = DM34ImageUtils.ndarray_to_imagedatadict(data)
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
                except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
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
        ParseDM34File.dm_write_header(file, file_version, ret)

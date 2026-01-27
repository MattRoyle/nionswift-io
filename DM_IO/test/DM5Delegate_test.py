import io
import unittest

import numpy

from DM_IO.test import DMDelegate_test
from DM_IO import DM5Delegate, DMDelegate
from nion.data import DataAndMetadata, Calibration


class TestDM5Delegate(DMDelegate_test.TestDMImportExportBase, unittest.TestCase):
    @property
    def dm_delegate(self) -> DMDelegate.DMIODelegate:
        return DM5Delegate.DM5IODelegate()

    @property
    def versions(self) -> list[int]:
        return [5]

    def test_metadata_write_read_round_trip(self) -> None: # This test is overridden since dm5 doesn't try to support tuples or nested lists
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
                    "three": [3, 4, 5], "threef": [3.0, 4.0, 5.0]
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

    def test_reference_images_load_properly(self) -> None: # This test is overridden until we put reference images in the resources folder
        pass
import unittest

from DM_IO.test import DMDelegate_test
from DM_IO import DM34Delegate, DMDelegate

class TestDM34ImportExport(DMDelegate_test.TestDMImportExportBase, unittest.TestCase):
    @property
    def dm_delegate(self) -> DMDelegate.DMIODelegate:
        return DM34Delegate.DM34IODelegate()

    @property
    def versions(self) -> list[int]:
        return [3, 4]
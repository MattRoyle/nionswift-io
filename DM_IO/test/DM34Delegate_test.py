import typing
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

    def assert_almost_equal(self, actual: typing.Any, desired: typing.Any, decimal: int = 7, err_msg: str | None = None) -> None:
        self.assertAlmostEqual(actual, desired, decimal, err_msg)

    def assert_equal(self, actual: typing.Any, desired: typing.Any, err_msg: str | None = None) -> None:
        self.assertEqual(actual, desired, err_msg)

    def assert_true(self, expr: typing.Any, err_msg: str | None = None) -> None:
        self.assertTrue(expr, err_msg)

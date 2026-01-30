import typing
import unittest

from DM_IO.test import DMDelegate_test
from DM_IO import DM5Delegate
from DM_IO import DMDelegate


class TestDM5Delegate(DMDelegate_test.TestDMImportExportBase, unittest.TestCase):
    @property
    def dm_delegate(self) -> DMDelegate.DMIODelegate:
        return DM5Delegate.DM5IODelegate()

    @property
    def versions(self) -> list[int]:
        return [5]

    def test_reference_images_load_properly(self) -> None:  # This test is overridden until we put reference images in the resources folder
        pass

    def assert_almost_equal(self, actual: typing.Any, desired: typing.Any, decimal: int = 7,
                            err_msg: str | None = None) -> None:
        self.assertAlmostEqual(actual, desired, decimal, err_msg)

    def assert_equal(self, actual: typing.Any, desired: typing.Any, err_msg: str | None = None) -> None:
        self.assertEqual(actual, desired, err_msg)

    def assert_true(self, expr: typing.Any, err_msg: str | None = None) -> None:
        self.assertTrue(expr, err_msg)

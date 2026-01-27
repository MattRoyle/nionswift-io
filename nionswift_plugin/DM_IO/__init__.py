"""
    Support for DM3 and DM4 I/O.
"""

# standard libraries
import gettext
import typing

# third party libraries
from nion.data import DataAndMetadata

# local libraries
from nionswift_plugin.DM_IO import dm3_image_utils
from nionswift_plugin.DM_IO import DMDelegates

_ = gettext.gettext

class DMIOExtension(object):

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.swift.extensions.dm3"

    def __init__(self, api_broker: typing.Any) -> None:
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__io_handler_ref = api.create_data_and_metadata_io_handler(DMDelegates.DM34IODelegate())
        self.__dm5_io_handler_ref = api.create_data_and_metadata_io_handler(DMDelegates.DM5IODelegate())

    def close(self) -> None:
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__io_handler_ref.close()
        self.__io_handler_ref = None

        self.__dm5_io_handler_ref.close()
        self.__dm5_io_handler_ref = None

# TODO: How should IO delegate handle title when reading using read_data_and_metadata

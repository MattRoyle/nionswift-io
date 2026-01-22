# ParseDM3File reads in a DM3 file and translates it into a dictionary
# this module treats that dictionary as an image-file and extracts the
# appropriate image data as numpy arrays.
# It also tries to create files from numpy arrays that DM can read.
#
# Some notes:
# Only complex64 and complex128 types are converted to structarrays,
# ie they're arrays of structs. Everything else, (including RGB) are
# standard arrays.
# There is a seperate DatatType and PixelDepth stored for images different
# from the tag file datatype. I think these are used more than the tag
# datratypes in describing the data.
# from .parse_dm3 import *

import array
import copy
import datetime
import pprint
import typing

import numpy
import numpy.typing
import pytz

from nion.data import Calibration
from nion.data import DataAndMetadata

from . import parse_dm3


def str_to_utf16_bytes(s: str) -> bytes:
    return s.encode('utf-16')

def get_datetime_from_timestamp_str(timestamp_str: str) -> typing.Optional[datetime.datetime]:
    if len(timestamp_str) in (23, 26):
        return datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
    elif len(timestamp_str) == 19:
        return datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
    return None

structarray_to_np_map = {
    ('d', 'd'): numpy.complex128,
    ('f', 'f'): numpy.complex64}

np_to_structarray_map = {v: k for k, v in iter(structarray_to_np_map.items())}

# we want to amp any image type to a single np array type
# but a sinlge np array type could map to more than one dm type.
# For the moment, we won't be strict about, eg, discriminating
# int8 from bool, or even unit32 from RGB. In the future we could
# convert np bool type eg to DM bool and treat y,x,3 int8 images
# as RGB.

# note uint8 here returns the same data type as int8 0 could be that the
# only way they're differentiated is via this type, not the raw type
# in the tag file? And 8 is missing!
dm_image_dtypes = {
    1: ("int16", numpy.int16),
    2: ("float32", numpy.float32),
    3: ("Complex64", numpy.complex64),
    6: ("uint8", numpy.int8),
    7: ("int32", numpy.int32),
    9: ("int8", numpy.int8),
    10: ("uint16", numpy.uint16),
    11: ("uint32", numpy.uint32),
    12: ("float64", numpy.float64),
    13: ("Complex128", numpy.complex128),
    14: ("Bool", numpy.int8),
    23: ("RGB", numpy.int32)
}


def imagedatadict_to_ndarray(imdict: dict[str, typing.Any]) -> numpy.typing.NDArray[typing.Any]:
    """
    Converts the ImageData dictionary, imdict, to an nd image.
    """
    arr = imdict['Data']
    im = None
    if isinstance(arr, array.array):
        im = numpy.asarray(arr, dtype=arr.typecode)
    elif isinstance(arr, parse_dm3.StructArray):
        t = typing.cast(tuple[str, str], tuple(arr.typecodes))
        im = numpy.frombuffer(
            typing.cast(typing.Any, arr.raw_data),  # huh?
            dtype=structarray_to_np_map[t])
    assert im is not None
    # print "Image has dmimagetype", imdict["DataType"], "numpy type is", im.dtype
    assert dm_image_dtypes[imdict["DataType"]][1] == im.dtype
    assert imdict['PixelDepth'] == im.dtype.itemsize
    im = im.reshape(imdict['Dimensions'][::-1])
    if imdict["DataType"] == 23:  # RGB
        im = im.view(numpy.uint8).reshape(im.shape + (-1, ))[..., :-1]  # strip A
        # NOTE: RGB -> BGR would be [:, :, ::-1]
    return im


def platform_independent_char(dtype: typing.Any) -> str:  # ugh dtype
    # windows and linux/macos treat dtype.char differently.
    # on linux/macos where 'l' has size 8, ints of size 4 are reported as 'i'
    # on windows where 'l' has size 4, ints of size 4 are reported as 'l'
    # this function fixes that issue.
    if dtype.char == 'l' and dtype.itemsize == 4: return 'i'
    if dtype.char == 'l' and dtype.itemsize == 8: return 'q'
    if dtype.char == 'L' and dtype.itemsize == 4: return 'I'
    if dtype.char == 'L' and dtype.itemsize == 8: return 'Q'
    return typing.cast(str, dtype.char)


def ndarray_to_imagedatadict(nparr: numpy.typing.NDArray[typing.Any]) -> dict[str, typing.Any]:
    """
    Convert the numpy array nparr into a suitable ImageList entry dictionary.
    Returns a dictionary with the appropriate Data, DataType, PixelDepth
    to be inserted into a dm3 tag dictionary and written to a file.
    """
    ret = dict[str, typing.Any]()
    dm_type = None
    for k, v in iter(dm_image_dtypes.items()):
        if v[1] == nparr.dtype.type:
            dm_type = k
            break
    if dm_type is None and nparr.dtype == numpy.uint8 and nparr.shape[-1] in (3, 4):
        ret["DataType"] = 23
        ret["PixelDepth"] = 4
        if nparr.shape[2] == 4:
            rgb_view = nparr.view(numpy.int32).reshape(nparr.shape[:-1])  # squash the color into uint32
        else:
            assert nparr.shape[2] == 3
            rgba_image = numpy.empty(nparr.shape[:-1] + (4,), numpy.uint8)
            rgba_image[:,:,0:3] = nparr
            rgba_image[:,:,3] = 255
            rgb_view = rgba_image.view(numpy.int32).reshape(rgba_image.shape[:-1])  # squash the color into uint32
        ret["Dimensions"] = list(rgb_view.shape[::-1])
        ret["Data"] = parse_dm3.DataChunkWriter(rgb_view)
    else:
        ret["DataType"] = dm_type
        ret["PixelDepth"] = nparr.dtype.itemsize
        ret["Dimensions"] = list(nparr.shape[::-1])
        if nparr.dtype.type in np_to_structarray_map:
            types = np_to_structarray_map[nparr.dtype.type]
            ret["Data"] = parse_dm3.StructArray(types)
            ret["Data"].raw_data = bytes(numpy.asarray(nparr).data)
        else:
            ret["Data"] = parse_dm3.DataChunkWriter(nparr)
    return ret


def display_keys(tag: dict[str, typing.Any]) -> None:
    tag_copy = copy.deepcopy(tag)
    for image_data in tag_copy.get("ImageList", list()):
        image_data.get("ImageData", dict()).pop("Data", None)
    tag_copy.pop("Page Behavior", None)
    tag_copy.pop("PageSetup", None)
    pprint.pprint(tag_copy)


def fix_strings(d: typing.Any) -> typing.Any:
    if isinstance(d, dict):
        r = dict()
        for k, v in d.items():
            if k != "Data":
                r[k] = fix_strings(v)
            else:
                r[k] = v
        return r
    elif isinstance(d, list):
        l = list()
        for v in d:
            l.append(fix_strings(v))
        return l
    elif isinstance(d, array.array):
        if d.typecode == 'H':
            return d.tobytes().decode("utf-16")
        else:
            return d.tolist()
    else:
        return d




# logging.debug(image_tags['ImageData']['Calibrations'])
# {u'DisplayCalibratedUnits': True, u'Dimension': [{u'Origin': -0.0, u'Units': u'nm', u'Scale': 0.01171875}, {u'Origin': -0.0, u'Units': u'nm', u'Scale': 0.01171875}, {u'Origin': 0.0, u'Units': u'', u'Scale': 0.01149425096809864}], u'Brightness': {u'Origin': 0.0, u'Units': u'', u'Scale': 1.0}}

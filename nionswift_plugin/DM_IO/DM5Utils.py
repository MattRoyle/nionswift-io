import datetime
import typing
import h5py
import numpy as np
from h5py import AttributeManager

DM_FILE_TYPES = np.ndarray | np.void | np.uint8 | np.uint16 | np.uint64 | np.uint32 | np.bytes_ | np.float32 | np.float64 | np.int16 | np.int32 | np.int64
DM_DICT_TYPES = typing.Tuple[int, ...] | int | str | typing.List[typing.Any] | float | bytes
EPOCH_AS_FILETIME = 116444736000000000  # January 1st 1970 as file time (hundreds of nanoseconds since January 1st 1601)
HUNDREDS_OF_NANOSECONDS = 10000000  # Hundreds of nanoseconds (0.1 microseconds) in a second


def get_datetime_from_filetime(filetime: int) -> datetime.datetime:
    """
    Converts a windows filetime to a datetime object
    Windows file time is: the time in hundreds of nanoseconds since January 1st 1601
    Convert to unix timestamp which is: the time in seconds since January 1st 1970
    """
    # Convert to epoch time in hundreds of nanoseconds, then convert to seconds by the division
    unix_seconds = (filetime - EPOCH_AS_FILETIME) / HUNDREDS_OF_NANOSECONDS
    return datetime.datetime.fromtimestamp(unix_seconds)


def get_filetime_from_datetime(time_dt: datetime.datetime) -> float:
    """
    Converts a datetime to a Windows file time
    """
    unix_seconds = time_dt.timestamp()
    return int(round(unix_seconds * HUNDREDS_OF_NANOSECONDS + EPOCH_AS_FILETIME))


def safe_create_group(base_group: h5py.Group, name: str) -> h5py.Group:
    group = base_group.get(name)
    if group is None:
        return base_group.create_group(name)
    return group


def data_serialization(data: DM_FILE_TYPES) \
        -> typing.Dict[str, DM_DICT_TYPES]:
    if isinstance(data, np.ndarray):
        serialized = {
            'data': data.tolist(),
            'dtype': str(type(data)),
            'shape': data.shape,
        }
    elif isinstance(data, np.void):
        serialized = {
            'data': data.tobytes(),
            'dtype': str(type(data)),
            'shape': data.shape,
        }
    elif isinstance(data, np.uint8) or isinstance(data, np.uint16) or isinstance(data, np.uint32) or isinstance(data, np.uint64):
        serialized = {
            'data': int(data),
            'dtype': str(type(data)),
        }
    elif isinstance(data, np.bytes_):
        serialized = {
            'data': data.decode('latin1'),  # latin1 has to be used in place of utf-8 because dm5 keys sometimes have non utf-8 bytes
            'dtype': str(type(data)),
        }
    elif isinstance(data, np.float32) or isinstance(data, np.float64):
        serialized = {
            'data': float(data),
            'dtype': str(type(data)),
        }
    elif isinstance(data, np.int16) or isinstance(data, np.int32) or isinstance(data, np.int64):
        serialized = {
            'data': int(data),
            'dtype': str(type(data)),
        }
    else:
        raise TypeError(f"{type(data)} is not supported.")
        serialized = {
            'data': data,
        }
    return serialized


def data_unserialization(serialized: typing.Mapping[str, DM_DICT_TYPES]) \
        -> DM_FILE_TYPES:
    shape = serialized.get('shape')
    dtype = serialized.get('dtype')
    data = serialized.get('data')
    return_data: DM_FILE_TYPES | None = None
    assert (data is not None)
    if shape is not None:
        if dtype == r"<class 'numpy.ndarray'>":
            data = typing.cast(typing.List[int | float], data)
            shape = typing.cast(typing.Tuple[int], shape)
            return_data = np.array(data).reshape(shape)

        elif dtype == r"<class 'numpy.void'>" and type(data) is bytes:
            data = typing.cast(bytes, data)
            return_data = np.void(data)
    elif dtype is not None:
        match dtype:
            case r"<class 'numpy.uint8'>":
                data = typing.cast(int, data)
                return_data = np.uint8(data)
            case r"<class 'numpy.uint16'>":
                data = typing.cast(int, data)
                return_data = np.uint16(data)
            case r"<class 'numpy.uint32'>":
                data = typing.cast(int, data)
                return_data = np.uint32(data)
            case r"<class 'numpy.uint64'>":
                data = typing.cast(int, data)
                return_data = np.uint64(data)
            case r"<class 'numpy.bytes_'>":
                data = typing.cast(str, data)
                return_data = np.bytes_(data.encode("latin1"))
            case r"<class 'numpy.float32'>":
                data = typing.cast(float, data)
                return_data = np.float32(data)
            case r"<class 'numpy.float64'>":
                data = typing.cast(float, data)
                return_data = np.float64(data)
            case r"<class 'numpy.int16'>":
                data = typing.cast(int, data)
                return_data = np.int16(data)
            case r"<class 'numpy.int32'>":
                data = typing.cast(int, data)
                return_data = np.int32(data)
            case r"<class 'numpy.int64'>":
                data = typing.cast(int, data)
                return_data = np.int64(data)
            case _:
                raise TypeError(f"{dtype!r} is not supported.")
    if return_data is not None:
        return return_data
    raise TypeError(f"{type(data)} is not supported.")


def convert_group_to_dict(group: h5py.Group) -> dict[str, typing.Any]:
    """
    Recursively visit all the nodes in the group. Converts groups to dicts, and stores attrs as a nested dict
    Datasets are currently ignored
    """
    def _convert_attrs_to_dict(attrs: AttributeManager) -> dict[str, typing.Any]:
        attrs_dict: dict[str, typing.Any] = dict()
        for key, value in attrs.items():
            value = data_serialization(value)
            if isinstance(key, bytes):
                key = key.decode('latin1')
            assert (isinstance(key, str))
            attrs_dict[key] = value

        return attrs_dict

    def _recursive_group_to_dict(base_group: h5py.Group) -> dict[str, typing.Any]:
        base_dict: dict[str, typing.Any] = dict()
        attributes = base_group.attrs
        if attributes is not None:
            base_dict['attrs'] = _convert_attrs_to_dict(attributes)
        for key, value in base_group.items():
            if isinstance(value, h5py.Group):
                base_dict[key] = _recursive_group_to_dict(value)
            elif isinstance(value, h5py.Dataset):
                pass
            else:
                raise TypeError
        return base_dict

    return _recursive_group_to_dict(group)


def convert_dict_to_group(base_dict: typing.Dict[str, typing.Any], group: h5py.Group) -> h5py.Group:
    """
    Converts dict to group by recursively going though the dicts values and rebuilding the groups, and attached attrs
    Datasets are currently ignored
    """
    def _convert_dict_to_attrs(attrs_dict: typing.Dict[str, typing.Any], base_group: h5py.Group) -> None:
        for key, value in attrs_dict.items():
            value = data_unserialization(value)
            assert (isinstance(key, str))
            if value:
                base_group.attrs.create(name=key, data=value)

    def _recursive_dict_to_group(recursive_dict: typing.Dict[str, typing.Any], top_group: h5py.Group) -> h5py.Group:
        for key, value in recursive_dict.items():
            if key == 'attrs':
                _convert_dict_to_attrs(value, top_group)
            elif isinstance(value, dict):
                new_group = safe_create_group(top_group, key)
                _recursive_dict_to_group(recursive_dict[key], new_group)
        return top_group

    return _recursive_dict_to_group(base_dict, group)

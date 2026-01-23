import datetime
import numbers
import typing
from types import MappingProxyType

import h5py
import numpy as np

NP_NUMERICAL_TYPES = np.uint8 | np.uint16 | np.uint64 | np.uint32 | np.float32 | np.float64 | np.int16 | np.int32 | np.int64
DM_FILE_TYPES = np.ndarray | np.void | np.bytes_ | NP_NUMERICAL_TYPES | np.bool_
VOID_FIELD_DICT_TYPES = str | int | dict[str, str | int]
DM_DICT_TYPES = typing.Tuple[int, ...] | int | str | typing.List[typing.Any] | float | bytes | dict[str, VOID_FIELD_DICT_TYPES]

FILETIME_TICKS_PER_MICROSECOND = 10 # Hundreds of nanoseconds in a microsecond
FILETIME_TICKS_PER_SECOND = 10000000  # Hundreds of nanoseconds (0.1 microseconds) in a second
FILETIME_EPOCH = datetime.datetime(1601, 1, 1, tzinfo=datetime.timezone.utc)

def get_datetime_from_filetime(filetime: int) -> datetime.datetime:
    """
    Converts a windows filetime to a datetime in UTC
    Windows file time is: the time in hundreds of nanoseconds since January 1st 1601 UTC
    """
    # Convert to epoch time in hundreds of nanoseconds, then convert to seconds by the division
    total_microseconds = filetime // FILETIME_TICKS_PER_MICROSECOND
    return FILETIME_EPOCH + datetime.timedelta(microseconds=total_microseconds)


def get_filetime_from_datetime(time_dt: datetime.datetime) -> int:
    """
    Converts a datetime to a Windows file time. If the datetime's timezone is None it is assumed to be UTC.
    """
    if time_dt.tzinfo is None:
        time_dt = time_dt.replace(tzinfo=datetime.timezone.utc)

    delta = time_dt.astimezone(datetime.timezone.utc) - FILETIME_EPOCH
    file_time_ticks = (delta.days * 24 * 3600 + delta.seconds) * FILETIME_TICKS_PER_SECOND + delta.microseconds * FILETIME_TICKS_PER_MICROSECOND
    return file_time_ticks


def get_or_create_group(base_group: h5py.Group, name: str) -> h5py.Group:
    group = base_group.get(name)
    if group is None:
        return base_group.create_group(name)
    return group


def convert_dm_to_swift(data: DM_FILE_TYPES | str | bool) \
        -> typing.Dict[str, DM_DICT_TYPES | dict[str, typing.Any]]:

    def serialize_dtype(fields: MappingProxyType[str, tuple[np.dtype[typing.Any], int]]) \
            -> dict[str, VOID_FIELD_DICT_TYPES]:
        void_dict: dict[str, VOID_FIELD_DICT_TYPES] = {}
        for name, (dtype, alignment) in fields.items():
            void_dict[name] = {
                'dtype': dtype.str,
                'alignment': alignment,
            }
        return void_dict

    if isinstance(data, (float, int)):
        serialized = data
    elif isinstance(data, (list, tuple)):
        serialized = {
            'data': data,
            'listtype': 'tuple' if type(data) is tuple else 'list'
        }
    elif isinstance(data, np.ndarray):
        serialized = {
            'data': [convert_dm_to_swift(x) for x in data.tolist()],
            'dtype': data.dtype.str,
            'shape': data.shape,
        }
    elif isinstance(data, np.void) and data.dtype.fields is not None:
        serialized = {
            'data': {
                'data': [convert_dm_to_swift(x) for x in data.tolist()],
                'fields': serialize_dtype(data.dtype.fields),
            },
            'dtype': data.dtype.str,
            'shape': data.shape,
        }
    elif isinstance(data, np.bytes_):
        serialized = {
            'data': data.decode('latin1'),  # latin1 has to be used in place of utf-8 because dm5 keys sometimes have non utf-8 bytes
            'dtype': data.dtype.str,
        }
    elif isinstance(data, (np.integer, np.bool_, np.floating)):
        serialized = {
            'data': data.item(),
            'dtype': data.dtype.str,
        }
    elif isinstance(data, str):
        serialized = {
            'data': data,
            'dtype': np.dtype(np.bytes_).str
        }
    elif isinstance(data, bool):
        serialized = {
            'data': data,
            'dtype': np.dtype(np.bool_).str
        }
    else:
        raise TypeError(f"{data}, {type(data)} is not supported.")
    return serialized


def data_deserialization(serialized: typing.Mapping[str, DM_DICT_TYPES]) \
        -> DM_FILE_TYPES:

    def deserialize_dtype(serialized_void: dict[str, VOID_FIELD_DICT_TYPES]) -> np.dtype:
        void_dict = {}
        for name, value in serialized_void.items():
            dtype_value = value.get('dtype')
            element_dtype = np.dtype(dtype_value)
            void_dict[name] = (element_dtype, value.get('alignment'))
        return np.dtype(void_dict)

    shape = serialized.get('shape')
    dtype = serialized.get('dtype', '')
    data = serialized.get('data')
    return_data: DM_FILE_TYPES | None = None
    np_dtype = np.dtype(dtype) if dtype else None
    assert data is not None
    if np.issubdtype(np_dtype, np.void):
        void_data = tuple(data.get('data'))
        void_fields = data.get('fields')
        data_dtype = deserialize_dtype(void_fields)
        return_data = np.void(void_data, data_dtype)
    elif np.issubdtype(np_dtype, np.ndarray):
        shape = typing.cast(typing.Tuple[int], shape)
        return_data = np.array(data).reshape(shape)
    elif np.issubdtype(np_dtype, np.bytes_):
        return_data = np.bytes_(data.encode("latin1"))
    elif np.issubdtype(np_dtype, np.bool_):
        return_data = np.bool_(data)
    elif np_dtype is not None:
        return_data = np.asarray(data, dtype=np_dtype)[()]
    if return_data is not None:
        return return_data
    raise TypeError(f"{dtype}, {shape} {data} {type(data)} is not supported.")


def convert_group_to_dict(group: h5py.Group) -> dict[str, typing.Any]:
    """
    Recursively visit all the nodes in the group. Converts groups to dicts, and stores attrs as a nested dict
    Datasets are currently ignored
    """
    def _convert_group_to_sequence(base_group: h5py.Group) -> list[typing.Any] | tuple[typing.Any, ...] | typing.Any:
        list_type_str = base_group.attrs.get('listtype')
        if list_type_str is not None:
            sequence = []
            data = base_group.attrs.get('data')
            if data is not None:
                if isinstance(data, np.ndarray):
                    for element in data:
                        if isinstance(element, h5py.Group):
                            sequence.append(_convert_group_to_sequence(element))
                        else:
                            sequence.append(element)
        else:
            sequence = base_group.attrs.get('data')

        return sequence if list_type_str == 'list' else tuple(sequence)

    def _convert_attrs_to_dict(attrs: h5py.AttributeManager) -> dict[str, typing.Any]:
        attrs_dict: dict[str, typing.Any] = dict()
        for key, value in attrs.items():
            value = convert_dm_to_swift(value)
            if isinstance(key, bytes):
                key = key.decode('latin1')
            assert (isinstance(key, str))
            attrs_dict[key] = value

        return attrs_dict

    def _recursive_group_to_dict(base_group: h5py.Group) -> dict[str, typing.Any]:
        base_dict: dict[str, typing.Any] = dict()
        attributes = base_group.attrs
        if attributes.get('listtype') is not None:
            return _convert_group_to_sequence(base_group)
        elif len(attributes.items()) != 0:
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
            if isinstance(value, dict):
                value = data_deserialization(value)
            elif isinstance(value, str):
                value = np.bytes_(value.encode())
            elif isinstance(value, bool):
                value = np.bool_(value)
            assert (isinstance(key, str))
            old_value = base_group.attrs.get(key)
            if old_value is not None and isinstance(old_value, (np.void, np.ndarray, np.generic)):
                if old_value.item() != value:   # Only update if it has changed
                    if type(old_value) == type(value):  # they have the same type
                        base_group.attrs[key] = value
                    elif (isinstance(old_value, np.floating) and isinstance(value, np.floating)) or (
                            isinstance(old_value, np.integer) and isinstance(value, np.integer)):  # If the old and new value are both the same numerical type besides the bytes
                        base_group.attrs[key] = value
                    elif isinstance(old_value, np.void) and isinstance(value, (tuple, list)):
                        base_group.attrs.modify(key, value)
            else:
                base_group.attrs.create(name=key, data=value)

    def _recursive_dict_to_group(recursive_dict: typing.Dict[str, typing.Any], top_group: h5py.Group) -> h5py.Group:
        for key, value in recursive_dict.items():
            if key == 'attrs':
                _convert_dict_to_attrs(value, top_group)
            elif isinstance(value, dict):
                new_group = get_or_create_group(top_group, key)
                _recursive_dict_to_group(recursive_dict[key], new_group)
            elif key is not None and key != '':
                if isinstance(value, str):
                    data = np.bytes_(value.encode())
                    top_group.attrs.create(name=key, data=data)
                elif isinstance(value, (list, tuple)):
                    new_group = get_or_create_group(top_group, key)
                    data = np.array([x for x in value if x is not None])
                    new_group.attrs.create(name='data', data=data)
                    new_group.attrs.create(name='listtype', data='list' if type(value) == list else 'tuple')
                elif isinstance(value, float):
                    top_group.attrs.create(name=key, data=np.float64(value))
                elif isinstance(value, int):
                    top_group.attrs.create(name=key, data=np.int64(value))
        return top_group

    return _recursive_dict_to_group(base_dict, group)


def squash_metadata_dict(metadata_dict: dict[str, typing.Any]) -> dict[str, typing.Any]:
    def _convert_attrs(attrs_dict: typing.Dict[str, typing.Any], base_dict: dict[str, typing.Any]) -> None:
        for key, value in attrs_dict.items():
            if isinstance(value, dict):
                data = value.get('data')  # Serialized attributes store the value at the key data
                if data is not None:
                    while isinstance(data, dict) and data.get('data') is not None:
                        data = data['data']  # The np.void data is another level deeper
                    value = data
            assert (isinstance(key, str))
            base_dict.update({key: value})

    def _recursive_squash_dict(base_dict: dict[str, typing.Any]) -> dict[str, typing.Any]:
        new_dict = {}
        for key, value in base_dict.items():
            if key == 'attrs':
                _convert_attrs(value, new_dict)
            elif isinstance(value, dict):
                new_dict[key] = _recursive_squash_dict(base_dict[key])
            elif isinstance(value, list):
                items = []
                for item in value:
                    if isinstance(item, dict):
                        items.append(_recursive_squash_dict(item))
                    elif isinstance(item, np.floating):
                        items.append(float(item))
                    elif isinstance(item, np.integer):
                        items.append(int(item))
                    else:
                        items.append(item)
                new_dict[key] = items
        return new_dict
    return _recursive_squash_dict(metadata_dict)

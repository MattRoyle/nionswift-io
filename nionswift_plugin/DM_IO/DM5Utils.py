import datetime
import typing
import h5py
import numpy as np
from h5py import AttributeManager

EPOCH_AS_FILETIME = 116444736000000000  # January 1st 1970 as file time (hundreds of nanoseconds since January 1st 1601)
HUNDREDS_OF_NANOSECONDS = 10000000 # Hundreds of nanoseconds (0.1 microseconds) in a second
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

def data_serialization(data: typing.Any) -> typing.Any:
    if isinstance(data, np.ndarray) or isinstance(data, np.void):
        serialized = {
            'data': data.tolist(),
            'dtype': str(type(data)),
            'shape': data.shape,
        }
    elif isinstance(data, np.uint8) or isinstance(data, np.uint16) or isinstance(data, np.uint32):
        serialized = {
            'data': int(data),
            'dtype': str(type(data)),
        }
    elif isinstance(data, np.bytes_):
        serialized = {
            'data': data.decode('latin1'),
            'dtype': str(type(data)),
        }
    elif isinstance(data, np.float32) or isinstance(data, np.float64):
        serialized = {
            'data': float(data),
            'dtype': str(type(data)),
        }
    else:
        serialized = {
            'data': data,
        }
    return serialized

def data_unserialization(serialized: typing.Any) -> typing.Any:
    shape = serialized.get('shape')
    dtype = serialized.get('dtype')
    data = serialized.get('data')
    if shape is not None:
        if dtype == 'np.ndarray':
            arr = np.array(data).reshape(shape)
            return arr
        elif dtype == 'nd.void':
            arr = np.void(data)
            return arr
    elif dtype is not None:
        match dtype:
            case 'np.uint8':
                data = data.astype(np.uint8)
            case 'np.uint16':
                data = data.astype(np.uint16)
            case 'np.uint32':
                data = data.astype(np.uint32)
            case 'np.bytes_':
                data = data.encode("latin1").astype(np.bytes_) # TODO does it work?
            case 'np.float32':
                data = data.astype(np.float32)
            case 'np.float64':
                data = data.astype(np.float64)
            case _:
                return data
        return data
    elif data is not None:
        return data

    return None

def convert_group_to_dict(group: h5py.Group) -> dict:

    def _convert_attrs_to_dict(attrs: AttributeManager) -> dict:
        attrs_dict = dict()
        for key, value in attrs.items():
            value = data_serialization(value)
            #print(key, type(key))
            if isinstance(key, bytes):
                key = key.decode('latin1') # latin1 has to be used in place of utf-8 because dm5 keys sometimes have non utf-8 bytes
            assert(isinstance(key, str))
            attrs_dict[key] = value

        return attrs_dict

    def _recursive_group_to_dict(base_group: h5py.Group) -> dict:
        base_dict = dict()
        attributes = base_group.attrs
        if attributes is not None:
            base_dict['attrs'] = _convert_attrs_to_dict(attributes)
        for key, value in base_group.items():
            if isinstance(value, h5py.Group):
                base_dict[key] = _recursive_group_to_dict(value)
            elif isinstance(value, h5py.Dataset):
                ds_str = value[()].tobytes().decode("latin1")  # Stores as ndarray
                base_dict[key] = dict()
                base_dict[key]['dm_dataset'] = {
                    'data': ds_str,
                    'name': value.name,
                    'dtype': value.dtype,
                    'ndim': value.ndim,
                    'shape': value.shape,
                    'maxshape': value.maxshape,
                    'chunks': value.chunks,
                    'compression': value.compression,
                    'shuffle': value.shuffle,
                    'fletcher32': value.fletcher32,
                    'scaleoffset': value.scaleoffset
                }
            else:
                raise TypeError
        return base_dict

    return _recursive_group_to_dict(group)


def convert_dict_to_group(base_dict: typing.Mapping[str, typing.Any], group: h5py.Group) -> h5py.Group:

    def _convert_dict_to_attrs(attrs_dict: dict, base_group: h5py.Group) -> None:
        for key, value in attrs_dict.items():
            value = data_unserialization(value)
            assert(isinstance(key, str))
            if value:
                base_group.attrs.create(name=key, data=value)

    def _recursive_dict_to_group(recursive_dict: typing.Mapping[str, typing.Any], top_group: h5py.Group) -> h5py.Group:
        for key, value in recursive_dict.items():
            if key == 'attrs':
                _convert_dict_to_attrs(value, top_group)
            elif isinstance(value, dict):
                if value.get('dm_dataset') is not None:
                    value = value.get('dm_dataset')
                    arr_buffer = value['data'].encode("latin1")
                    ndarr = np.frombuffer(arr_buffer, dtype=value['dtype'])
                    print(ndarr)
                    dataset = ndarr.reshape(value['shape'])
                    group_dataset = top_group.get(value['name'])
                    print(group_dataset)
                    group_dataset = top_group.create_dataset(name=value['name'],
                                                             data=dataset) if group_dataset is None else group_dataset
                    print(group_dataset)
                else:
                    new_group = safe_create_group(top_group, key)
                    _recursive_dict_to_group(recursive_dict[key], new_group)
        return top_group

    return _recursive_dict_to_group(base_dict, group)



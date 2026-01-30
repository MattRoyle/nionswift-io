import json
import typing
import types

import h5py
import numpy

NP_NUMERICAL_TYPES = numpy.uint8 | numpy.uint16 | numpy.uint64 | numpy.uint32 | numpy.float32 | numpy.float64 | numpy.int16 | numpy.int32 | numpy.int64
DM_FILE_TYPES = numpy.ndarray | numpy.void | numpy.bytes_ | NP_NUMERICAL_TYPES | numpy.bool_
VOID_FIELD_DICT_TYPES = str | int | dict[str, str | int]
DM_DICT_TYPES = typing.Tuple[int, ...] | int | str | typing.List[typing.Any] | float | bytes | dict[str, VOID_FIELD_DICT_TYPES]

FieldInfo: typing.TypeAlias = typing.Union[ tuple[numpy.dtype[typing.Any], int], tuple[numpy.dtype[typing.Any], int, typing.Any]]

# h5py utility functions

def get_or_create_group(base_group: h5py.Group, name: str) -> h5py.Group:
    """Creates a group if one doesn't already exist and returns it or return the existing group."""
    group = base_group.get(name)
    if group is None:
        return base_group.create_group(name)
    return group


def save_attr_to_group(name: str, data: typing.Any, group: h5py.Group,
                       dtype: NP_NUMERICAL_TYPES | list[tuple[str, ...]] | None = None) -> None:
    """Save data to a group's attribute, converting python types to numpy types."""
    value: DM_FILE_TYPES | None = None
    if isinstance(data, str):
        value = numpy.bytes_(data.encode())
    elif isinstance(data, bool):
        value = numpy.bool_(data)
    elif isinstance(data, float):
        if dtype is None:
            value = numpy.float64(data)
        elif numpy.issubdtype(dtype, numpy.floating):
            value = numpy.asarray(data, dtype=dtype)[()]
    elif isinstance(data, int):
        if dtype is None:
            value = numpy.int64(data)
        elif numpy.issubdtype(dtype, numpy.integer):
            value = numpy.asarray(data, dtype=dtype)[()]
    elif isinstance(data, (tuple, list)) and dtype is not None and not isinstance(dtype, NP_NUMERICAL_TYPES):
        np_dtype = numpy.dtype(dtype)
        value = numpy.void(data, np_dtype)

    if value is not None:
        if group.attrs.get(name) is not None:
            group.attrs[name] = value
        else:
            group.attrs.create(name, value)
    else:
        raise TypeError(f"{data}, {type(data)} is not supported.")

# Serialization Functions

def serialize_dm_attrs_into_swift_metadata(data: DM_FILE_TYPES | int | float) \
        -> typing.Dict[str, DM_DICT_TYPES | dict[str, typing.Any]] | int | float:
    """Converts data in dm5 attrs, DM_FILE_TYPES, into a dict with the data at dict['data'] in a type that can be stored
    in swift metadata.

    Information to rebuild the original type is stored in the dict with the converted data.
    Data can be int or float when there is a recursive call in the ndarray or void, otherwise it is a DM_FILE_TYPE.
    This will raise a TypeError if the type of data was not in DM_FILE_TYPES as an unhandled case.
    """

    def serialize_void_dtype_into_swift_metadata(
            fields: typing.Optional[typing.Mapping[str, FieldInfo]]) \
            -> dict[str, VOID_FIELD_DICT_TYPES]:

        void_dict: dict[str, VOID_FIELD_DICT_TYPES] = {}
        if fields is None:
            return void_dict

        for name, info in fields.items():
            dtype, alignment = info[0], info[1]
            void_dict[name] = {
                'dtype': dtype.str,
                'alignment': alignment,
            }
        return void_dict

    serialized: typing.Dict[str, DM_DICT_TYPES | dict[str, typing.Any]]
    if isinstance(data, numpy.ndarray):
        serialized = {
            'data': [serialize_dm_attrs_into_swift_metadata(x) for x in data.tolist()],
            'dtype': data.dtype.str,
            'shape': data.shape,
        }
    elif isinstance(data, numpy.void) and data.dtype.fields is not None:
        serialized = {
            'data': {
                'data': [serialize_dm_attrs_into_swift_metadata(x) for x in data.tolist()],
                'fields': serialize_void_dtype_into_swift_metadata(data.dtype.fields),
            },
            'dtype': data.dtype.str,
            'shape': data.shape,
        }
    elif isinstance(data, numpy.bytes_):
        serialized = {
            'data': data.decode('latin1'),  # latin1 has to be used in place of utf-8 because dm5 keys sometimes have non utf-8 bytes
            'dtype': data.dtype.str,
        }
    elif isinstance(data, (numpy.integer, numpy.bool_, numpy.floating)):
        serialized = {
            'data': data.item(),
            'dtype': data.dtype.str,
        }
    elif isinstance(data, (float, int)):
        return data
    else:
        raise TypeError(f"{data}, {type(data)} is not supported.")
    return serialized


def deserialize_dm_attrs_from_swift_metadata(serialized: typing.Mapping[str, DM_DICT_TYPES]) \
        -> DM_FILE_TYPES:
    """Convert the swift metadata back to dm5 attrs data that was serialized using serialize_dm_attrs_into_swift_metadata.

    Uses the stored information in the dict about the original type, and then converts the data to be that type again.
    This will raise an exception if the dictionary passed was not one of the possible serialized versions from serialize_dm_attrs_into_swift_metadata.
    """

    def deserialize_dtype(serialized_void: dict[str, VOID_FIELD_DICT_TYPES]) -> numpy.dtype | None:
        print("this ran")
        void_dict = []
        for name, value in serialized_void.items():
            if isinstance(value, dict):
                dtype_value = value.get('dtype')
                if isinstance(dtype_value, str):
                    element_dtype = numpy.dtype(dtype_value)
                    void_dict.append((element_dtype, value.get('alignment')))
        void_type = numpy.dtype(void_dict) if void_dict else None
        if isinstance(void_type, numpy.dtype):
            return void_type
        return None

    shape = serialized.get('shape')
    dtype = serialized.get('dtype', '')
    data = serialized.get('data')
    return_data: DM_FILE_TYPES | None = None
    np_dtype = numpy.dtype(dtype) if dtype else None
    assert data is not None
    if numpy.issubdtype(np_dtype, numpy.void):
        assert isinstance(data, dict)
        void_data = (data.get('data'),)
        void_fields = data.get('fields', dict())
        data_dtype = deserialize_dtype(void_fields)
        return_data = numpy.void(void_data, data_dtype)
    elif numpy.issubdtype(np_dtype, numpy.ndarray):
        shape = typing.cast(typing.Tuple[int], shape)
        return_data = numpy.array(data).reshape(shape)
    elif numpy.issubdtype(np_dtype, numpy.bytes_):
        return_data = numpy.bytes_(data.encode("latin1"))
    elif numpy.issubdtype(np_dtype, numpy.bool_):
        return_data = numpy.bool_(data)
    elif numpy.issubdtype(np_dtype, numpy.integer) or numpy.issubdtype(np_dtype, numpy.floating):
        return_data = numpy.asarray(data, dtype=np_dtype)[()]

    if return_data is not None:
        return return_data
    raise TypeError(f"{dtype}, {shape} {data} {type(data)} is not supported.")


def squash_metadata_dict(metadata_dict: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Removes the stored types made by serialize_dm_attrs_into_swift_metadata leaving only the data in the swift metadata dict.

    This is done so the swift metadata structure for everything but dm_metadata is identical to what would be seen in dm3/4.
    """

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
                    elif isinstance(item, numpy.floating):
                        items.append(float(item))
                    elif isinstance(item, numpy.integer):
                        items.append(int(item))
                    else:
                        items.append(item)
                new_dict[key] = items
        return new_dict

    return _recursive_squash_dict(metadata_dict)

# h5py Groups to dictionaries

def convert_group_to_dict(group: h5py.Group) -> dict[str, typing.Any]:
    """Converts h5py groups to a dict, with nested groups becoming nested dicts, and stores attrs as a nested dict, ignores datasets.

    Recursively visit all the nodes in the group converting attrs data to a type that can be stored in swift metadata DM_DICT_TYPES.
    """

    def _convert_void_to_sequence(data: numpy.void, np_dtype: numpy.dtype, meta: dict[str, str]) \
            -> list[typing.Any] | tuple[typing.Any, ...] | typing.Any:
        """Recursively convert a void to a list or tuple, using metadata dictionary to determine the container type"""
        if np_dtype.fields is None:
            return data.item()

        sequence: list[typing.Any] = []
        container = meta.get("container", "list")
        for i, name in enumerate(np_dtype.names):
            sub_dt = dict(np_dtype.fields).get(name)[0]
            field_val = data[name]

            if sub_dt.fields is not None:
                child_meta = meta.get('children', [])[i]
                sequence.append(_convert_void_to_sequence(field_val, sub_dt, child_meta))
            else:
                if isinstance(field_val, numpy.generic):
                    sequence.append(field_val.item())
                else:
                    sequence.append(field_val)

        return sequence if container == "list" else tuple(sequence)

    def _convert_attrs_to_dict(attrs: h5py.AttributeManager) -> dict[str, typing.Any]:
        """Converts group attributes into a dict, serializing the data from dm5 types"""
        attrs_dict: dict[str, typing.Any] = dict()
        for key, value in attrs.items():
            if isinstance(key, bytes):  # Some of the keys were encoded so this was required
                key = key.decode('latin1')
            assert (isinstance(key, str))
            if '.__meta__' in key:
                continue
            if isinstance(value, numpy.void) and key + '.__meta__' in attrs.keys():
                meta_value = attrs[key + '.__meta__']
                if isinstance(meta_value, (numpy.bytes_, bytes, bytearray)):
                    meta_json = bytes(meta_value).decode()
                else:
                    meta_json = str(meta_value)
                meta = json.loads(meta_json)
                value = _convert_void_to_sequence(value, value.dtype, meta)
            else:
                value = serialize_dm_attrs_into_swift_metadata(value)

            attrs_dict[key] = value

        return attrs_dict

    def _recursive_group_to_dict(base_group: h5py.Group) -> dict[str, typing.Any]:
        base_dict: dict[str, typing.Any] = dict()
        attributes = base_group.attrs
        if len(attributes.items()) != 0:
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
    """Converts a dict to group going though any nested dicts rebuilding the groups, and attached attrs. Datasets are currently ignored.

    If a nested dict named attrs is encountered then it will use that to create attributes for the parent group
    """
    def _convert_dict_to_attrs(attrs_dict: typing.Dict[str, typing.Any], base_group: h5py.Group) -> None:
        for key, value in attrs_dict.items():
            if isinstance(value, dict):
                value = deserialize_dm_attrs_from_swift_metadata(value)
            elif isinstance(value, str):
                value = numpy.bytes_(value.encode())
            elif isinstance(value, bool):
                value = numpy.bool_(value)
            base_group.attrs.create(name=key, data=value)

    def _convert_sequence_to_void(data: list[typing.Any] | tuple[typing.Any, ...] | typing.Any) -> tuple[numpy.dtype | None, (DM_FILE_TYPES | None) | tuple[DM_FILE_TYPES, ...]]:
        """Converts a tuple/list to a form that can be stored in a numpy.void. Returns numpy.dtype, and a tuple of values to use to create the void.

        Stores a recursive dictionary of the original container type (list or tuple) in the dtype metadata so it the original data can be rebuilt from the numpy void.
        """
        if not isinstance(data, (list, tuple)):
            np_dtype = None
            if isinstance(data, numpy.generic):
                np_dtype = data.dtype
            elif isinstance(data, bool):
                np_dtype = numpy.dtype(numpy.bool_)
            elif isinstance(data, int):
                np_dtype = numpy.min_scalar_type(data)
            elif isinstance(data, float):
                np_dtype = numpy.result_type(data)
            elif isinstance(data, str):
                np_dtype = numpy.dtype(f"U{len(data)}")  # Array-protocol str from numpy
            elif isinstance(data, bytes):
                np_dtype = numpy.dtype(f"S{len(data)}")

            if np_dtype is not None:
                return np_dtype, data  # base case of recursion
            else:
                return None, None

        # The data is a list or tuple
        fields: typing.List[typing.Tuple[str, typing.Any]] = []
        values: typing.List[typing.Any] = []
        children_meta: typing.List[dict] = []

        for i, element in enumerate(data):
            name = f"f{i}"
            np_dtype, value = _convert_sequence_to_void(element)
            if value is None:
                continue  # Don't store unsupported types
            fields.append((name, np_dtype))
            values.append(value)
            if np_dtype.metadata and "container" in np_dtype.metadata:
                children_meta.append({
                    "container": np_dtype.metadata["container"],
                    "children": np_dtype.metadata.get("children", [])
                })
            else:
                children_meta.append({"container": "scalar"})

        container_type = "list" if isinstance(data, list) else "tuple"
        metadata = {"container": container_type, "children": children_meta}

        dt = numpy.dtype(fields, metadata=metadata)
        value = tuple(values)

        return dt, value

    def _recursive_dict_to_group(recursive_dict: typing.Dict[str, typing.Any], top_group: h5py.Group) -> h5py.Group:
        for key, value in recursive_dict.items():
            if key is None or key == '':  # ensure the key is a valid name for h5py
                continue
            if key == 'attrs':
                _convert_dict_to_attrs(value, top_group)
            elif isinstance(value, dict):
                new_group = get_or_create_group(top_group, key)
                _recursive_dict_to_group(recursive_dict[key], new_group)
            elif isinstance(value, str):
                data = numpy.bytes_(value.encode())
                top_group.attrs.create(name=key, data=data)
            elif isinstance(value, (list, tuple)):
                np_dtype, data = _convert_sequence_to_void(value)
                # dtype metadata is not saved by h5py so instead the metadata is saved as an attribute named '__meta__'
                meta = dict(np_dtype.metadata) or {"container": "scalar", "children": []}
                top_group.attrs.create(name=key + ".__meta__", data=json.dumps(meta))

                if data:
                    data = numpy.array(data, dtype=np_dtype)[()]
                else:  # If the data is an empty tuple h5py throws an error unless it is done without the dtype
                    data = numpy.array(data)[()]
                top_group.attrs.create(name=key, data=data)
            elif isinstance(value, float):
                top_group.attrs.create(name=key, data=numpy.float64(value))
            elif isinstance(value, int):
                top_group.attrs.create(name=key, data=numpy.int64(value))
        return top_group

    return _recursive_dict_to_group(base_dict, group)

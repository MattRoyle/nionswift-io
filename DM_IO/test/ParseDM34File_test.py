import io
import typing
import array
import numpy
import unittest

from DM_IO import ParseDM34File, DM34ImageUtils


def is_equal(r: typing.Any, data: typing.Any) -> bool:
    # we use this to compare the read and written data
    if isinstance(r, (list, tuple)):
        return len(r) == len(data) and all(is_equal(x, y) for x, y in zip(r, data))
    elif isinstance(r, dict):
        return r.keys() == data.keys() and all(is_equal(r[k], data[k]) for k in r)
    elif isinstance(r, array.array) and isinstance(data, ParseDM34File.DataChunkWriter):
        return numpy.array_equal(numpy.array(r), data.data)
    else:
        return bool(r == data)


class ParseDM34FileTest(unittest.TestCase):

    def check_write_then_read_matches(self, data: typing.Any, write_func: typing.Callable[..., typing.Any], read_func: typing.Callable[..., typing.Any], _assert: bool = True) -> typing.Any:
        # we confirm that reading a written element returns the same value
        s = io.BytesIO()
        header = write_func(s, data)
        s.seek(0)
        if header is not None:
            r, hy = read_func(s)
        else:
            r = read_func(s)
        if _assert:
            self.assertTrue(is_equal(r, data))
        return r

    def test_dm_read_struct_types(self) -> None:
        s = io.BytesIO()
        types = [2, 2, 2]
        ParseDM34File.dm_write_struct_types(s, types)
        s.seek(0)
        in_types, headerlen = ParseDM34File.dm_read_struct_types(s)
        self.assertEqual(in_types, types)

    def test_simpledata(self) -> None:
        self.check_write_then_read_matches(45, ParseDM34File.dm_write_types[ParseDM34File.get_dmtype_for_name('long')], ParseDM34File.dm_read_types[ParseDM34File.get_dmtype_for_name('long')])
        self.check_write_then_read_matches(2**30, ParseDM34File.dm_write_types[ParseDM34File.get_dmtype_for_name('uint')], ParseDM34File.dm_read_types[ParseDM34File.get_dmtype_for_name('uint')])
        self.check_write_then_read_matches(34.56, ParseDM34File.dm_write_types[ParseDM34File.get_dmtype_for_name('double')], ParseDM34File.dm_read_types[ParseDM34File.get_dmtype_for_name('double')])

    def test_read_string(self) -> None:
        data = "MyString"
        ret = self.check_write_then_read_matches(data, ParseDM34File.dm_write_types[ParseDM34File.get_dmtype_for_name('array')], ParseDM34File.dm_read_types[ParseDM34File.get_dmtype_for_name('array')], False)
        self.assertEqual(data, DM34ImageUtils.fix_strings(ret))

    def test_array_simple(self) -> None:
        dat = ParseDM34File.DataChunkWriter(numpy.array([0] * 256, dtype=numpy.int8))
        self.check_write_then_read_matches(dat, ParseDM34File.dm_write_types[ParseDM34File.get_dmtype_for_name('array')], ParseDM34File.dm_read_types[ParseDM34File.get_dmtype_for_name('array')])

    def test_array_struct(self) -> None:
        dat = ParseDM34File.StructArray(['h', 'h', 'h'])
        dat.raw_data = array.array('b', [0, 0] * 3 * 8)  # two bytes x 3 'h's x 8 elements
        self.check_write_then_read_matches(dat, ParseDM34File.dm_write_types[ParseDM34File.get_dmtype_for_name('array')], ParseDM34File.dm_read_types[ParseDM34File.get_dmtype_for_name('array')])

    def test_tagdata(self) -> None:
        for d in [45, 2**30, 34.56, ParseDM34File.DataChunkWriter(numpy.array([0] * 256, dtype=numpy.int8))]:
            self.check_write_then_read_matches(d, ParseDM34File.dm_write_tag_data, ParseDM34File.dm_read_tag_data)

    def test_tagroot_dict(self) -> None:
        mydata = dict[str, typing.Any]()
        self.check_write_then_read_matches(mydata, ParseDM34File.dm_write_tag_root, ParseDM34File.dm_read_tag_root)
        mydata = {"Bob": 45, "Henry": 67, "Joe": 56}
        self.check_write_then_read_matches(mydata, ParseDM34File.dm_write_tag_root, ParseDM34File.dm_read_tag_root)

    def test_tagroot_dict_complex(self) -> None:
        mydata = {"Bob": 45, "Henry": 67, "Joe": {
                  "hi": [34, 56, 78, 23], "Nope": 56.7, "d": ParseDM34File.DataChunkWriter(numpy.array([0] * 32, dtype=numpy.uint16))}}
        self.check_write_then_read_matches(mydata, ParseDM34File.dm_write_tag_root, ParseDM34File.dm_read_tag_root)

    def test_tagroot_list(self) -> None:
        # note any strings here get converted to 'H' arrays!
        mydata = list[int]()
        self.check_write_then_read_matches(mydata, ParseDM34File.dm_write_tag_root, ParseDM34File.dm_read_tag_root)
        mydata = [45,  67,  56]
        self.check_write_then_read_matches(mydata, ParseDM34File.dm_write_tag_root, ParseDM34File.dm_read_tag_root)

    def test_struct(self) -> None:
        # note any strings here get converted to 'H' arrays!
        mydata = tuple[typing.Any]()
        self.check_write_then_read_matches(mydata, ParseDM34File.dm_write_types[ParseDM34File.get_dmtype_for_name('struct')], ParseDM34File.dm_read_types[ParseDM34File.get_dmtype_for_name('struct')])
        mydata = (3, 4, 56.7)
        self.check_write_then_read_matches(mydata, ParseDM34File.dm_write_types[ParseDM34File.get_dmtype_for_name('struct')], ParseDM34File.dm_read_types[ParseDM34File.get_dmtype_for_name('struct')])

    def test_image(self) -> None:
        im = numpy.random.randint(low=0, high=65536, size=(32,), dtype=numpy.uint16)
        im_tag = {"Data": ParseDM34File.DataChunkWriter(im),
                  "Dimensions": [23, 45]}
        s = io.BytesIO()
        ParseDM34File.dm_write_tag_root(s, im_tag)
        s.seek(0)
        ret = ParseDM34File.dm_read_tag_root(s)
        self.assertTrue(is_equal(ret["Data"], im_tag["Data"]))
        self.assertEqual(im_tag["Dimensions"], ret["Dimensions"])

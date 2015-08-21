#!/usr/bin/env python

# Copyright 2014 Camiolog Inc.
# Massimo Di Pierro and Luca de Alfaro
# BSD License.

import base64
import datetime
import importlib
import json
import numpy
import unittest

class Storage(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class Serializable(object):

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def to_json(self, pack_ndarray=True, tolerant=True, indent=None):
        return Serializable.dumps(self, pack_ndarray=pack_ndarray, tolerant=tolerant, indent=indent)

    @staticmethod
    def dumps(obj, pack_ndarray=True, tolerant=True, indent=None):
        """This function dumps an object to extended json.
        All objects that are json-serializable data types, and in addition,
        numpy float, bool, or int types are serialized.
        If pack_ndarray is True, all numpy arrays are also serialized.
        In addition, all classes that extend Serializable are also serialized.
        If tolerant is True (the default), to_json does not complain if the
        objects to be serialized contain non-serializable attributes
        (for instance, objects of generic classes that do not extend
        Serializable).  If tolerant is False, an exception of class
        ValueError will be raised when non-serializable attributes are found.
        Setting tolerant=True can be used for conditionally serializing only
        portions of an object.

        """
        def custom(o):
            if isinstance(o, Serializable):
                module = o.__class__.__module__.split('campil.')[-1]
                d = {'__type__': '%s.%s' % (o.__class__.__module__,
                                              o.__class__.__name__)}
                d.update(item for item in o.__dict__.items() if not item[0].startswith('_'))
                return d
            elif isinstance(o, datetime.datetime):
                d = {'__type__': 'datetime.datetime',
                     'date': o.isoformat()}
                return d
            elif type(o) == numpy.float:
                return float(o)
            elif type(o) == numpy.bool:
                return bool(o)
            elif isinstance(o, (numpy.int, numpy.int16, numpy.int32, numpy.int8)):
                return int(o)
            elif type(o).__module__ == numpy.__name__:
                if pack_ndarray and isinstance(o, numpy.ndarray):
                    d = {'__type__': '%s.ndarray' % numpy.__name__,
                         'dtype': str(o.dtype),
                         'shape': o.shape,
                         'data': base64.b64encode(o.tostring())}
                    return d
                else:
                    return '<numpy %r>' % (o.shape, )
            elif isinstance(o, (int, long, str, unicode, float, bool, list, tuple, dict)):
                return o
            elif tolerant:
                return None
            else:
                raise ValueError("Cannot encode in json object %r" % o)
        return json.dumps(obj, default=custom, indent=indent)

    @staticmethod
    def from_json(s, objectify=True, remapper={}, fallback={}):
        """
        Decodes a generalized json object.
        remapper is an optional dictionary that specifies a translation from old
        to new class names.
        fallback is an optional dictionary that specifies a mapping from class names,
        to the modules where the class declarations can be found.  This can be used if
        the code is reorganized, so that classes that used to be defined in a module
        can be found if they are moved into a new module.
        """
        def hook(o):
            meta_module, meta_class = None, o.get('__type__')
            if meta_class == 'datetime.datetime':
                try:
                    tmp = datetime.datetime.strptime(
                        o['date'], '%Y-%m-%dT%H:%M:%S.%f')
                except Exception, e:
                    tmp = datetime.datetime.strptime(
                        o['date'], '%Y-%m-%dT%H:%M:%S')
                return tmp
            elif meta_class == '%s.ndarray' % numpy.__name__:
                data = base64.b64decode(o['data'])
                dtype = o['dtype']
                shape = o['shape']
                v = numpy.frombuffer(data, dtype=dtype)
                v = v.reshape(shape)
                return v

            elif meta_class and '.' in meta_class:
                # correct for classes that have migrated from one module to another
                meta_class = remapper.get(meta_class, meta_class)
                # separate the module name from the actual class name
                meta_module, meta_class = meta_class.rsplit('.',1)

            if meta_class is not None:
                del o['__type__']
                # this option is for backward compatibility, in case classes
                # change the module where they can be found.
                if meta_class in fallback:
                    meta_module = fallback.get(meta_class)

                if meta_module is not None and objectify:
                    prefix = ''
                    module = importlib.import_module(prefix+meta_module)
                    cls = getattr(module, meta_class)
                    obj = cls()
                    obj.__dict__.update(o)
                    o = obj
            elif type(o).__name__ == 'dict':
                o = Storage(o)
            return o

        return json.loads(s, object_hook=hook)

    @staticmethod
    def loads(s):
        return Serializable.from_json(s)

class TestSerializable(unittest.TestCase):

    def test_simple(self):
        a = Serializable()
        a.x = 1
        a.y = 'test'
        a.z = 3.14
        b = Serializable.from_json(a.to_json())
        self.assertEqual(a, b)

    def test_datetime(self):
        a = Serializable()
        a.x = datetime.datetime(2015,1,3)
        b = Serializable.from_json(a.to_json())
        self.assertEqual(a, b)

    def test_recursive(self):
        a = Serializable()
        a.x = Serializable()
        a.x.y = 'test'
        b = Serializable.from_json(a.to_json())
        self.assertEqual(a, b)

    def test_numpy(self):
        a = Serializable()
        a.x = numpy.array([[1,2,3],[4,5,6]], dtype=numpy.int32)
        b = Serializable.from_json(a.to_json(pack_ndarray=True))
        self.assertEqual(numpy.sum(numpy.abs(a.x - b.x)), 0)

if __name__ == '__main__':
    unittest.main()

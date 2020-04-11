import six
import copy
import numpy as np
import pandas as pd 
from collections import MutableMapping
pd.set_option("display.precision", 4)

class ADict(dict, MutableMapping):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # to prevent overwrite of internal attributes by new keys
        # see _valid_name()
        self._setattr('_allow_invalid_attributes', False)

    def _build(self, obj, **kwargs):
        """
        We only want dict like elements to be treated as recursive AttrDicts.
        """
        return obj

    # --- taken from AttrDict

    def __getstate__(self):
        return self.copy(), self._allow_invalid_attributes

    def __setstate__(self, state):
        mapping, allow_invalid_attributes = state
        self.update(mapping)
        self._setattr('_allow_invalid_attributes', allow_invalid_attributes)

    @classmethod
    def _constructor(cls, mapping):
        return cls(mapping)

    # --- taken from MutableAttr

    def _setattr(self, key, value):
        """
        Add an attribute to the object, without attempting to add it as
        a key to the mapping (i.e. internals)
        """
        super(MutableMapping, self).__setattr__(key, value)

    def __setattr__(self, key, value):
        """
        Add an attribute.
        key: The name of the attribute
        value: The attributes contents
        """
        if self._valid_name(key):
            self[key] = value
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__setattr__(key, value)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute creation.".format(
                    cls=self.__class__.__name__
                )
            )

    def _delattr(self, key):
        """
        Delete an attribute from the object, without attempting to
        remove it from the mapping (i.e. internals)
        """
        super(MutableMapping, self).__delattr__(key)

    def __delattr__(self, key, force=False):
        """
        Delete an attribute.
        key: The name of the attribute
        """
        if self._valid_name(key):
            del self[key]
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__delattr__(key)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute deletion.".format(
                    cls=self.__class__.__name__
                )
            )

    def __call__(self, key):
        """
        Dynamically access a key-value pair.
        key: A key associated with a value in the mapping.
        This differs from __getitem__, because it returns a new instance
        of an Attr (if the value is a Mapping object).
        """
        if key not in self:
            raise AttributeError(
                "'{cls} instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    def __getattr__(self, key):
        """
        Access an item as an attribute.
        """
        if key not in self or not self._valid_name(key):
            raise AttributeError(
                "'{cls}' instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    @classmethod
    def _valid_name(cls, key):
        """
        Check whether a key is a valid attribute name.
        A key may be used as an attribute if:
         * It is a string
         * The key doesn't overlap with any class attributes (for Attr,
            those would be 'get', 'items', 'keys', 'values', 'mro', and
            'register').
        """
        return (
                isinstance(key, six.string_types) and
                not hasattr(cls, key)
        )

class Gridnet(ADict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(args[0], self.__class__):
            net = args[0]
            self.clear()
            self.update(**net.deepcopy())

    def deepcopy(self):
        return copy.deepcopy(self)

    def __repr__(self):  # pragma: no cover
        r = "This Gridapower network includes the following parameter tables:"
        par = []
        res = []
        for tb in list(self.keys()):
            if not tb.startswith("_") and isinstance(self[tb], pd.DataFrame) and len(self[tb]) > 0:
                if 'res_' in tb:
                    res.append(tb)
                else:
                    par.append(tb)
        for tb in par:
            length = len(self[tb])
            r += "\n   - %s (%s %s)" % (tb, length, "elements" if length > 1 else "element")
        if res:
            r += "\n and the following results tables:"
            for tb in res:
                length = len(self[tb])
                r += "\n   - %s (%s %s)" % (tb, length, "elements" if length > 1 else "element")
        return r


# By stackoverflow user "Steve Zelaznik"
# from https://stackoverflow.com/a/25332884
# CC BY-SA 3.0


def col(i):
    ''' For binding named attributes to spots inside subclasses of tuple.'''
    g = tuple.__getitem__
    @property
    def _col(self):
        return g(self,i)
    return _col

class Item(tuple):
    ''' Designed for storing key-value pairs inside
        a FrozenDict, which itself is a subclass of frozenset.
        The __hash__ is overloaded to return the hash of only the key.
        __eq__ is overloaded so that normally it only checks whether the Item's
        key is equal to the other object, HOWEVER, if the other object itself
        is an instance of Item, it checks BOTH the key and value for equality.

        WARNING: Do not use this class for any purpose other than to contain
        key value pairs inside FrozenDict!!!!

        The __eq__ operator is overloaded in such a way that it violates a
        fundamental property of mathematics. That property, which says that
        a == b and b == c implies a == c, does not hold for this object.
        Here's a demonstration:
            [in]  >>> x = Item(('a',4))
            [in]  >>> y = Item(('a',5))
            [in]  >>> hash('a')
            [out] >>> 194817700
            [in]  >>> hash(x)
            [out] >>> 194817700
            [in]  >>> hash(y)
            [out] >>> 194817700
            [in]  >>> 'a' == x
            [out] >>> True
            [in]  >>> 'a' == y
            [out] >>> True
            [in]  >>> x == y
            [out] >>> False
    '''

    __slots__ = ()
    key, value = col(0), col(1)
    def __hash__(self):
        return hash(self.key)
    def __eq__(self, other):
        if isinstance(other, Item):
            return tuple.__eq__(self, other)
        return self.key == other
    def __ne__(self, other):
        return not self.__eq__(other)
    def __str__(self):
        return '%r: %r' % self
    def __repr__(self):
        return 'Item((%r, %r))' % self

class FrozenDict(frozenset):
    ''' Behaves in most ways like a regular dictionary, except that it's immutable.
        It differs from other implementations because it doesn't subclass "dict".
        Instead it subclasses "frozenset" which guarantees immutability.
        FrozenDict instances are created with the same arguments used to initialize
        regular dictionaries, and has all the same methods.
            [in]  >>> f = FrozenDict(x=3,y=4,z=5)
            [in]  >>> f['x']
            [out] >>> 3
            [in]  >>> f['a'] = 0
            [out] >>> TypeError: 'FrozenDict' object does not support item assignment

        FrozenDict can accept un-hashable values, but FrozenDict is only hashable if its values are hashable.
            [in]  >>> f = FrozenDict(x=3,y=4,z=5)
            [in]  >>> hash(f)
            [out] >>> 646626455
            [in]  >>> g = FrozenDict(x=3,y=4,z=[])
            [in]  >>> hash(g)
            [out] >>> TypeError: unhashable type: 'list'

        FrozenDict interacts with dictionary objects as though it were a dict itself.
            [in]  >>> original = dict(x=3,y=4,z=5)
            [in]  >>> frozen = FrozenDict(x=3,y=4,z=5)
            [in]  >>> original == frozen
            [out] >>> True

        FrozenDict supports bi-directional conversions with regular dictionaries.
            [in]  >>> original = {'x': 3, 'y': 4, 'z': 5}
            [in]  >>> FrozenDict(original)
            [out] >>> FrozenDict({'x': 3, 'y': 4, 'z': 5})
            [in]  >>> dict(FrozenDict(original))
            [out] >>> {'x': 3, 'y': 4, 'z': 5}   '''

    __slots__ = ()
    def __new__(cls, orig={}, **kw):
        if kw:
            d = dict(orig, **kw)
            items = map(Item, d.items())
        else:
            try:
                items = map(Item, orig.items())
            except AttributeError:
                items = map(Item, orig)
        return frozenset.__new__(cls, items)

    def __repr__(self):
        cls = self.__class__.__name__
        items = frozenset.__iter__(self)
        _repr = ', '.join(map(str,items))
        return '%s({%s})' % (cls, _repr)

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        diff = self.difference
        item = diff(diff({key}))
        key, value = set(item).pop()
        return value

    def get(self, key, default=None):
        if key not in self:
            return default
        return self[key]

    def __iter__(self):
        items = frozenset.__iter__(self)
        return map(lambda i: i.key, items)

    def keys(self):
        items = frozenset.__iter__(self)
        return map(lambda i: i.key, items)

    def values(self):
        items = frozenset.__iter__(self)
        return map(lambda i: i.value, items)

    def items(self):
        items = frozenset.__iter__(self)
        return map(tuple, items)

    def copy(self):
        cls = self.__class__
        items = frozenset.copy(self)
        dupl = frozenset.__new__(cls, items)
        return dupl

    @classmethod
    def fromkeys(cls, keys, value):
        d = dict.fromkeys(keys,value)
        return cls(d)

    def __hash__(self):
        kv = tuple.__hash__
        items = frozenset.__iter__(self)
        return hash(frozenset(map(kv, items)))

    def __eq__(self, other):
        if not isinstance(other, FrozenDict):
            try:
                other = FrozenDict(other)
            except Exception:
                return False
        return frozenset.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

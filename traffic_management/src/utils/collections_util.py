from collections import namedtuple


def named_tuple_from_dict(name, dict_):

    Return = namedtuple(name, ' '.join(dict_.keys()))

    return Return(*dict_.values())


# Code adapted from stackoverflow.com/questions/3318625/how-to-implement-an-efficient-bidirectional-hash-table

class bidict(dict):

    def __init__(self, *args, **kwargs):

        super(bidict, self).__init__(*args, **kwargs)

        # initialize inverse dict
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):

        # remove old reference
        if key in self:
            self.inverse[self[key]].remove(key)

            # clean remaining empty list
            if self[key] in self.inverse and not self.inverse[self[key]]:
                del self.inverse[self[key]]

        # insert references
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):

        value = self[key]

        # remove references
        self.inverse.setdefault(value, []).remove(key)
        super(bidict, self).__delitem__(key)

        # clean remaining empty list
        if value in self.inverse and not self.inverse[value]:
            del self.inverse[value]


# Code adapted from stackoverflow.com/questions/1151658/python-hashable-dicts

class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self))

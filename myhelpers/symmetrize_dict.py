import collections


def symmetrize_dict(indict):
    outdict = {key: value for key, value in indict.iteritems()}
    for key, value in indict.iteritems():
        if isinstance(key, collections.Iterable):
            outdict[key[::-1]] = value
    return outdict


def unsymmetrize_dict(indict):
    outdict = {}
    avoidset = set()
    for key, value in indict.iteritems():
        if key not in avoidset:
            if isinstance(key, collections.Iterable):
                outdict[key] = value
                avoidset.add(key[::-1])
    return outdict


def unsymmetrize_list(inlist):
    avoidset = set()
    outlist = []
    for value in inlist:
        if value not in avoidset:
            if isinstance(value, collections.Iterable):
                outlist.append(value)
                avoidset.add(value[::-1])
    return outlist

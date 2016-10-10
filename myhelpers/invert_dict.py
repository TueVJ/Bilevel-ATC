def invert_dict(d, to_list=False):
    '''
        Converts a many-to-few dictionary into a few-to-sets-of-many dictionary
        Input:
            d: Dictionary to be inverted ({k: v, ...})
            to_list: Bool, if True, convert sets to lists before returning.
        Output:
            out: Inverted dictionary keyed by values ({v: set((k, ...)), ...})
    '''
    from collections import defaultdict
    out = defaultdict(set)
    for k, v in d.iteritems():
        out[v].add(k)
    if to_list:
        out = {k: list(v) for k, v in out.iteritems()}
    return out

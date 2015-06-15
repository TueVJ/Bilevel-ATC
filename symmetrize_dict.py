import collections

def symmetrize_dict(indict):
	outdict = {key:value for key, value in indict.iteritems()}
	for key,value in indict.iteritems():
		if isinstance(key, collections.Iterable):
			outdict[key[::-1]] = value
	return outdict

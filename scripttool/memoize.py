"""
memoize.py: module providing a memoize file cache decorator for methods

before making function calls, set memoize.readcache to True for reading from cache,
if memoize.readcache is False than the cache is only written.

customize memoize.dbname if necessary, the default is /tmp/memoizedb
note that a hash value may be appended to the given database name

Example:
@memoize.filecache
def squareroot(x):
    return sqrt(x)

2011-06-30, Steffen Waldherr
"""

import shelve
import cPickle as pickle
import hashlib
import marshal

memoizeconfig = {"readcache": False}
dbname = "/tmp/memoizedb"

def filecache(func):
    funchash = hashlib.sha256(marshal.dumps(func.func_code)).hexdigest()
    def callf(*args, **kwargs):
        try:
            arghash = pickle.dumps((args,kwargs))
        except (TypeError, pickle.PickleError) , e:
            print "Warning: %s in memoizing %s, not using cache. (%s)" % (type(e).__name__, func.__name__, str(e))
            funcres = func(*args, **kwargs)
        else:
            cache = shelve.open(dbname + funchash)
            try:
                if not memoizeconfig["readcache"] or arghash not in cache:
                    funcres = func(*args, **kwargs)
                    try:
                        cache[arghash] = funcres
                    except pickle.PickleError as e:
                        print "Warning: %s in memoizing %s, not storing in cache. (%s)" % (type(e).__name__, func.__name__, str(e))
                else:
                    print "Reading %s results from cache %s ..." % (func.__name__, funchash)
                    funcres = cache[arghash]
            finally:
                cache.close()
        return funcres
    callf.func_doc = func.func_doc
    return callf
        
def set_config(**kwargs):
    """
    set configuration, e.g.
    
    set_config(readcache=True) to enable reading from cache
    """
    memoizeconfig.update(kwargs)

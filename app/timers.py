import os
import time
from datetime import datetime


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        # _args = [str(arg)
        #          for arg in args] + [f'{key} = {kw[key]}' for key in kw]
        if f.__name__ != 'db_exec' or os.getenv('log_db_exec'):
            print('[%s] %r %s took: %2.4f sec' %
                  (str(datetime.utcnow()).split(".")[0],
                   f.__module__, f.__name__, te - ts))
        return result

    return timed


def atimeit(f):
    async def timed(*args, **kw):
        ts = time.time()
        result = await f(*args, **kw)
        te = time.time()
        # _args = [str(arg)
        #          for arg in args] + [f'{key} = {kw[key]}' for key in kw]
        if f.__name__ != 'db_exec' or os.getenv('log_db_exec'):
            print('[%s] %r %s took: %2.4f sec' %
                  (str(datetime.utcnow()).split(".")[0],
                   f.__module__, f.__name__, te - ts))
        return result

    return timed

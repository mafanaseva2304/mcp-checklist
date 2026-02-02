import pickle
import dpath
from math import log as ln  # ln используется в формулах
from math import sqrt as sqrt  # sqrt используется в формулах
from math import exp as exp  # exp используется в формулах


async def rec_run(session, req, data=None):
    return await session.execute(req, data)


def calc_f(f: str):
    return eval(f)


def unpickle(blob):
    return pickle.loads(blob)


def unpickle_l(blob):
    return pickle.load(blob)

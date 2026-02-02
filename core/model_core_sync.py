from __future__ import annotations
import json
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Union, TYPE_CHECKING
from core.utils.shortcuts import unpickle as unpickle_l
import requests

import pandas as pd
from psycopg2 import extras

import sys

from psycopg2.sql import SQL

if TYPE_CHECKING:
    from psycopg2.extensions import connection
    from typing import List, Union


def cur_execute(cur, query, params):
    cur.execute(query, params)
    # extras.execute_values(cur, query, params)


def cur_execute_fetch(cur, query, params):
    cur.execute(query, params)


def unpickle(module):
    return unpickle_l(module)


def cur_post(url, payload, timeout):
    return requests.post(url, json=payload, timeout=timeout)


def cur_get(url):
    return requests.get(url)

# -*- coding: utf-8 -*-
"""
R language helper functions
"""

import gc
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface import RRuntimeError

def get_robject(result, obj_name):
    """
    Rのリストオブジェクトから指定した名前のオブジェクトを取得する
    """
    for i in result.items():
        if i[0] == obj_name:
            return i[1]

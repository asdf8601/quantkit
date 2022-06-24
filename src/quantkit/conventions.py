"""Conventions module."""
import numpy as np
import pandas as pd
from typing import Union


ArrayLike = Union[np.array, pd.Series, pd.DataFrame]
ReducedOut = Union[np.array, pd.Series, pd.DataFrame, float]


BYEAR = 261

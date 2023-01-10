from typing import Union
import numpy as np


def scale(x) -> Union[float, np.array]:
    return x / 2500 * 255

import numpy as np
import re
from typing import List, Tuple, Dict, Set, Union, Optional

class EinopsError(Exception):
    """Error raised by einops operations"""
    pass

def _product(sequence):
    """
    Calculate product of a sequence of numbers
    Implementation taken as it is from minimalist product in einop/einops.py
    """
    if len(sequence) == 0:
        return 1
    result = 1
    for element in sequence:
        result *= element
    return result
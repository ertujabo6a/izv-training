#!/usr/bin/env python3
# coding=utf-8
"""
IZV Training
Autor: Aleksandr Dmitriev (240259/xdmitr01)
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any

def integrate(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, steps=1000) -> float:
    # Generate points on a linear distribution from 'a' to 'b'
    x_values = np.linspace(a, b, steps)

    # Calculate the function values at all points
    y_values = f(x_values)

    # Width of one rectangle (step)
    dx = (b - a) / steps

    # Calculate the integral using the rectangle method
    integral_value = np.sum(y_values) * dx

    return integral_value

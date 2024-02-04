#!/usr/bin/env python3
"""
Script for automatic testing

Execution:
   pytest
or
   python3 -m pytest
"""
import part01
import numpy as np
import os
import pytest


def test_integrate():
    """Test calculation of the integral"""
    def f(x): return 10 * x + 2
    r = part01.integrate(f, 0, 1, 100)
    assert r == pytest.approx(7)


def test_generate_fn():
    "Test generating graphs with multiple functions"
    part01.generate_graph([1., 2., -2.], show_figure=False,
                          save_path="tmp_fn.png")
    assert os.path.exists("tmp_fn.png")


def test_generate_sin():
    "Test generating graphs with sine functions"
    part01.generate_sinus(show_figure=False, save_path="tmp_sin.png")
    assert os.path.exists("tmp_sin.png")


def test_download():
    "Test data download"
    data = part01.download_data()

    assert len(data) == 40
    assert data[0]["position"] == "Cheb"
    assert data[0]["lat"] == pytest.approx(50.0683)
    assert data[0]["long"] == pytest.approx(12.3913)
    assert data[0]["height"] == pytest.approx(483.0)

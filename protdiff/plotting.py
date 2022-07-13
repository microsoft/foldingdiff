"""
Utility functions for plotting
"""


import os

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
assert os.path.isdir(PLOT_DIR)

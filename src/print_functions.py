import numpy as np
from IPython.display import Latex, display
import array_to_latex as a2l
import ipywidgets as ipw
import pandas as pd
from numpy.polynomial import Polynomial as P
from IPython.display import Markdown as md


def print_matrix(np_array_matrix):
    """
    Parameters:
    np_array_matrix - 2 dimensional ndarray.

    Prints the input matrix.
    """
    np_array_matrix = np_array_matrix.round(3)
    if np.isreal(np_array_matrix).all():
        np_array_matrix = np_array_matrix.real
    latex_code = a2l.to_ltx(
        np_array_matrix, frmt='{:.3}', arraytype='pmatrix', print_out=False)
    n = latex_code.rfind('\n')
    latex_code = latex_code[:n] + '\t' + latex_code[n+1:]
    return latex_code


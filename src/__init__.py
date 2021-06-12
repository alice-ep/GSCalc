import ipywidgets as widgets
import numpy as np
from ipywidgets import GridspecLayout, Layout, interact, fixed
from IPython.display import display, Latex, HTML, Markdown
from .Gram_Schmidt import InnerProdSpace as GS
import array_to_latex as a2l

gs = None
inprod_mat = None

# Widgets
style = {'description_width': 'initial'}
n_select = widgets.Dropdown(
    options=[('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5)],
    value=2,
    description='Choose n:', style=style,
)

m_select = widgets.Dropdown(
    options=[('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5)],
    value=2,
    description='Choose number of vectors:', style=style,
)

use_matrix = widgets.Checkbox(
    value=False,
    description='Use a custom inner product?', style=style,
    disabled=False
)

out = widgets.Output()
def print_matrix(np_array_matrix):
    np_array_matrix = np_array_matrix.round(3)
    if np.isreal(np_array_matrix).all():
        np_array_matrix = np_array_matrix.real
    latex_code = a2l.to_ltx(
        np_array_matrix, frmt='{:.3}', arraytype='pmatrix', print_out=False)
    n = latex_code.rfind('\n')
    latex_code = latex_code[:n] + '\t' + latex_code[n+1:]
    return latex_code

# Create input widgets
def init_input(n_dim, m_dim):
    display(create_grid(n_dim, m_dim))
    # display(use_matrix)
    interact(create_A, n_dim=fixed(n_dim), use=use_matrix)

def create_grid(n_dim, m_dim):
    global gs

    n_old, m_old = 0, 0
    if gs is not None:
        n_old, m_old = gs.n_rows, gs.n_columns
        old_grid = gs
    gs = GridspecLayout(n_dim+1, m_dim, layout=Layout(
        width=f'{m_dim*90}px', height='auto'))
    
    for j in range(m_dim):
        gs[0, j] = widgets.Label(f'$v_{j+1}$')
    for i in range(1, n_dim+1):
        for j in range(m_dim):
            if i < n_old and j < m_old:
                gs[i, j] = old_grid[i, j]
            else:
                gs[i, j] = widgets.FloatText(
                    layout=Layout(width='80px', height='auto'))
    
    return gs

def create_A(n_dim, use):
    global inprod_mat

    if not use:
        # Empty grid - don't display anything if the checkbox isn't marked
        return GridspecLayout(1, 1)

    n_old = 0
    if inprod_mat is not None:
        n_old = inprod_mat.n_rows
        old_grid = inprod_mat
    inprod_mat = GridspecLayout(n_dim+1, n_dim, layout=Layout(
        width=f'{n_dim*90}px', height='auto'))
    
    inprod_mat[0,:] = widgets.Label(f'Input matrix A:')
    for i in range(1,n_dim+1):
        for j in range(n_dim):
            if i < n_old and j < n_old-1:
                inprod_mat[i, j] = old_grid[i, j]
            else:
                inprod_mat[i, j] = widgets.FloatText(
                    layout=Layout(width='80px', height='auto')) 
    return inprod_mat

# Display functionality
def grid_to_mat(grid, n, m, A=False):
    mat = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            mat[i,j] = grid[i+1,j].value
    return mat


def display_result(b):
    # clear output
    out.clear_output()

    # load matrix
    n, m = n_select.value, m_select.value
    gs_values = grid_to_mat(gs, n, m)
    gs_values = gs_values.T.tolist()
    
    A_mat = None
    basis_string = "w.r.t. the standard inner product"
    if use_matrix.value:
        A_mat = grid_to_mat(inprod_mat, n, n, True)
        A_mat = A_mat.tolist()
        basis_string = "w.r.t. the inner product $\\langle v, u \\rangle = v^T A u$"
        
    try:
        grams = GS(n_select.value, A=A_mat)
    except ValueError:
        with out:
            display(Latex("Matrix passed is not positive definite, try again"))
        b.description = 'try again'
        return
    
    result = grams.orthonormalize(gs_values)
    if result.shape[0] == 0:
        with out:
            display(Latex("Please enter at least one non-zero vector"))
        b.description = 'try again'
        return
    is_basis = (result.shape[1] == n_select.value)
    
    # redraw output
    with out:
        if is_basis:
            display(Latex(f'The following is an orthonormal basis for $\\mathbb{{R}}^{n_select.value}$ ' + basis_string))
        else:
            display(Latex(f'The following is not a basis for $\\mathbb{{R}}^{n_select.value}$ ' + basis_string))
        
        latex_vecs = r'\begin{Bmatrix}'
        for i in range(result.shape[1]):
            latex_vecs += print_matrix((result[:,i]).reshape(-1,1))
        latex_vecs += r'\end{Bmatrix}'
        display(Latex(f'{latex_vecs}'))

    b.description = 'recalculate'
    return


def start():
    display(HTML(r'<style> body { direction: ltr; } .widget-label { max-width:350ex; text-align:center} </style>'))
    out.clear_output()

    widgets.interact(init_input, n_dim=n_select, m_dim=m_select)
    button = widgets.Button(description="Calculate", icon='calculator')
    display(button)
    display(out)
    button.on_click(display_result)

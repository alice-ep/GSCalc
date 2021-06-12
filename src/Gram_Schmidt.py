import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Latex
import abc


class InnerProdSpace:
    def __init__(self, n_dim, A=None):
        self._n_dim = n_dim
        if isinstance(A, list):
            A = np.array(A)
        if not isinstance(A, np.ndarray):
            A = np.eye(n_dim)
        ev = np.linalg.eig(A)[0]
        if ev.min() <= 0:
            raise ValueError("Matrix passed isn't PD")
        self.A = A

    @property
    def n_dim(self):
        return self.n_dim

    def prod(self, u, v) -> float:
        u, v = (np.array(u)).reshape(-1, 1), (np.array(v)).reshape(-1, 1)
        return float(u.T @ self.A @ v)

    def norm(self, u) -> float:
        return np.sqrt(self.prod(u, u))

    def orthonormalize(self,  vectors: list):
        vectors = [np.array(vec, dtype=np.float) for vec in vectors]
        orth_vectors = []
        for i, v in enumerate(vectors):
            orthogonal_to_previous = v.copy()
            for prev_vec in orth_vectors:
                orthogonal_to_previous -= self.prod(v, prev_vec) * prev_vec
                # print(f'Substracting {prev_vec} from {v} resulting in {orthogonal_to_previous}')
            if not np.allclose(orthogonal_to_previous, np.zeros_like(orthogonal_to_previous)):
                orthonormal_to_previous = orthogonal_to_previous / self.norm(orthogonal_to_previous)
                orth_vectors.append(orthonormal_to_previous)
        return np.asarray(orth_vectors).T
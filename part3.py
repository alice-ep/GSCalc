import numpy as np
from Gram_Schmidt import InnerProdSpace

V = np.array([[1,0,0,2],[2,-1,0,1],[2, -1, 0 ,1], [0,0,3,3]])

s = InnerProdSpace(4, np.eye(4,4))
res = s.orthonormalize(np.transpose(V))

Q = np.transpose(res)

print(Q)

#<p,q> = p(0)q(0) + p(2)q(2) + p(-1)q(-1)

A = np.array([[3,1,5],[1,5,7],[5,7,17]])

V = np.eye(3,3)

s = InnerProdSpace(3, A)
res = s.orthonormalize(np.transpose(V))
    
Q = np.transpose(res)

print(Q)



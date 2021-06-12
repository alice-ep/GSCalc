import numpy as np
import sys
from Gram_Schmidt import InnerProdSpace
#insert project imports here

tests = 100000

max_n = 5

limits = [(-1,1),(-100,100),(20,40),(-11111,11111)]



threshold = 0.0001

def orthonormal(A, Q):
    gram = np.matmul(np.matmul(np.transpose(Q),A),Q)
    expected = np.eye(gram.shape[0],gram.shape[0])
    diff = np.abs(gram - expected)
    return np.sum(np.sum(diff)) < threshold
    
def same_column_space(V,Q):
    for v in np.transpose(V):
        if not dependent(Q,v):
            return False
            
    for q in np.transpose(Q):
        if not dependent(V,q):
            return False
            
    return True
    
        
def dependent(V,y):
    x = np.linalg.lstsq(V,y,rcond=None)[0]
    diff = np.abs(y - np.matmul(V,x))
    return np.sum(diff) < threshold
    
        
for i in range(tests):
    num_min = limits[i%len(limits)][0]
    num_max = limits[i%len(limits)][1]

    n = np.random.randint(2,max_n)
    m = np.random.randint(1,n)
    
    
    V = (num_max - num_min) * np.random.sample((n,m)) + num_min
    
    A_inited = False
    
    A = 0
    
    while not A_inited:
        A = (num_max - num_min) * np.random.sample((n,n)) + num_min
        A = np.matmul(np.transpose(A),A)
        
        eig = np.linalg.eig(A)[0]
        
        if min(eig) > 0:
            A_inited = True
            
    s = InnerProdSpace(n, A)
    res = s.orthonormalize(np.transpose(V))
    
    Q = np.transpose(res)
    
    if not orthonormal(A, Q) or not same_column_space(V,Q):
        print(n,m)
        print(V)
        print(Q)
        sys.exit(0)
        


        
print("success")
        
    
    
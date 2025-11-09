import numpy as np
def c_hst(U,V):
    d = U.shape[0]
    return 1 - 1/(d**2)*(np.abs(np.trace(np.conjugate(np.transpose(V))@U)))**2
def f_hat(U,V):
    d = U.shape[0]
    return 1 - d/(d + 1) * c_hst(U,V)
def frobenius(U,V):
    return 1/2*np.linalg.norm(V - U,'fro')
def frobenius_fidelity(U,V):
    d = U.shape[0]
    return 1 - d/(d + 1) + 1/(d*(d+1))*(d - frobenius(U,V))**2
def f2(U, V):   
    d = U.shape[0]
    return np.abs((np.trace(np.conjugate(np.transpose(V))@U))/d)**2
# u = np.random.rand(16, 16)
# q, r = np.linalg.qr(u)
# import time
# start = time.time()
# q = np.identity(4)
# print(f2(q,q))
# # np.trace(np.conjugate(np.transpose(q))@q)
# end = time.time()
# print(end - start)
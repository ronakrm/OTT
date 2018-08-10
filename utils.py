import numpy as np

from pymanopt import Problem
from pymanopt.manifolds import Product

def makeProductManifold():
	return NotImplemented

def aOTTreconst(core_list, d, n, r, out_dims):
	U = []
	start = 0
	end = 0
	for i in range(0, d):
		end = end + n[i]
		tmp = np.stack(core_list[start:end], axis=1)
		start = end
		if r[i+1] > r[i]:
			U.append( np.transpose(tmp, [2, 1, 0]) )
		else:
			U.append( tmp )

	W = U[0] # first
	for i in range(1, d): # second through last
		W = np.tensordot(W, U[i], axes=1)
	W = np.reshape(W, out_dims)

	return W

def OTTreconst(core_list, d, n, r, out_dims):
	n = [0] + n
	U = []
	for i in range(0, d):
		U.append( np.reshape(core_list[i], [r[i], n[i+1], r[i+1]]) )
	U[d-1] = np.einsum('abc,c->abc', U[d-1], core_list[d])

	W = U[0] # first
	for i in range(1, d): # second through last
		W = np.tensordot(W, U[i], axes=1)
	W = np.reshape(W, out_dims)
	
	return W
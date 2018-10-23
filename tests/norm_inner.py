import numpy as np
import matplotlib.pyplot as plt
r = 10000
norms = np.zeros([1000,1])
for i in range(0,1000):
	a = np.random.randn(1,r)
	a = a/np.linalg.norm(a)
	b = np.random.randn(1,r)
	b = b/np.linalg.norm(b)
	#if np.abs(np.dot(a,b.T))>1:
	norms[i] = np.dot(a,b.T)

plt.hist(norms)
plt.show()
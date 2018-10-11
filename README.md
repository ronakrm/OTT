# OTT
Orthogonal Tensor Trains

sOTT is the square, SO(n) version. Variables are stored in vectors, mapped to a skew symmetric matrix, and transformed into the orthogonal cores.

aOTT is the approximate orthogonal representation. Each core of the TT is represented by a similar sized orthogonal matrix, which is orthogonal by initialization and projection via gradStep in stiefel_ops.py

sOTT/aOTT should have the same interfaces, so to test differences it should only require replacing the variable and doing a stiefel update as needed for aOTT. 

tfTensor.py files are strict tensors, without operator/multiplication definitions.
Variable.py files are -operators- in the sense that the TT has an input dimension and an output dimension. The mult function computes matrix multiplication with the tensor representation without explicit blow up of the full tensor.

Q() functions are the variable/core representations. U() functions are the full tensor train representations, and W() functions are the matricized versions of the operators.

stiefel_ops.py is taken largely from pymanopt.

lq_other.py is a simple test of aOTT.
lq_sott.py is a simple test of sOTT.

mnist_ott.py is the simple classification task for MNIST with 1 -OTT variable layer.

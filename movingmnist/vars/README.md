Tensor and Variable definitions for TT and OTT representations.

TT is the standard tensor-train representation.

aOTT is the tensor-train representation but with each r by r slice as a separate variable, allowing for separate Stiefel manifold optimization to be performed on each.

sOTT is the efficient, Square Stiefel/SO(r) representation in which each slice of a core is explicitly represented via a mapping from an r(r-1)/2 vector to an orthogonal r by r matrix.

These details can be found in the paper linked in the main README.

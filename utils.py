import numpy as np


def spiral_cw(A):
    A = np.array(A)
    out = []
    while (A.size):
        out.append(A[0])  # take first row
        A = A[1:].T[::-1]  # cut off first row and rotate counterclockwise
    return np.concatenate(out)


def base_spiral(nrow, ncol):
    return spiral_cw(np.arange(nrow * ncol).reshape(nrow, ncol))[::-1]


def to_spiral(A):
    A = np.array(A)
    B = np.empty_like(A)
    B.flat[base_spiral(*A.shape)] = A.flat
    return B

def from_spiral(A):
    A = np.array(A)
    return A.flat[base_spiral(*A.shape)].reshape(A.shape)

# cython: embedsignature=True
# cython: initializedcheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

from cython.parallel import prange
import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float64
ITYPE = np.intp

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

from cython.parallel import prange

cdef DTYPE_t INF = np.inf


from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

cdef class DTWDistance():
    cdef inline DTYPE_t dist(self, const DTYPE_t* x1, const DTYPE_t* x2, const ITYPE_t size_x1, const ITYPE_t size_x2) nogil except -1:
        cdef:
            DTYPE_t *C = <DTYPE_t *> malloc(size_x1 * size_x2  * sizeof(DTYPE_t))
            DTYPE_t dist = 0
            ITYPE_t i, j
        try:
            for i in range(size_x1):
                for j in range(size_x2):
                    dist = (x1[i] - x2[j])**2
                    if i == 0 and j == 0:
                        C[i * size_x2 + j] = dist
                        continue
                    if i == 0 and j != 0:
                        C[i * size_x2 + j] = dist + C[i * size_x2 + (j - 1)]
                        continue
                    if i != 0 and j == 0:
                        C[i * size_x2 + j] = dist + C[(i - 1) * size_x2 + j]
                        continue
                    C[i * size_x2 + j] = dist + min(C[(i - 1) * size_x2 + j], C[i * size_x2 + (j - 1)], C[(i - 1) * size_x2 + (j - 1)])
            return sqrt(C[size_x1 * size_x2 - 1])
        finally:
            free(C)

    cdef int cdist(self, const DTYPE_t[:, ::1] X, const DTYPE_t[:, ::1] Y,
                   DTYPE_t[:, ::1] D) nogil except -1:
        cdef ITYPE_t i1, i2
        for i1 in range(X.shape[0]):
            for i2 in range(Y.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &Y[i2, 0], X.shape[1], Y.shape[1])
        return 0

    cdef int parallel_cdist(self, const DTYPE_t[:, ::1] X, const DTYPE_t[:, ::1] Y,
                   DTYPE_t[:] L, const DTYPE_t eps, const ITYPE_t n_jobs) nogil except -1:
        cdef:
            ITYPE_t n_samples = Y.shape[0]
            ITYPE_t i, this_n, end, start
            ITYPE_t i1, i2, num_threads
            ITYPE_t chunk_size = Y.shape[0] // n_jobs
            DTYPE_t dist
        start = 0
        num_threads = n_jobs
        if chunk_size == 0:
            chunk_size = n_samples
            num_threads = 1
        for i in prange(0, n_samples, chunk_size, num_threads = n_jobs, nogil=True):
            start = i
            end = min(i + chunk_size, n_samples)
            for i1 in range(X.shape[0]):
                for i2 in range(end-start):
                    dist = self.dist(&X[i1, 0], &Y[start+i2, 0], X.shape[1], Y.shape[1])
                    if dist < eps:
                        L[i1] += 1
        return 0

    cdef int parallel_cdist1(self, const DTYPE_t[:, ::1] X, const DTYPE_t[:, ::1] Y,
                   DTYPE_t[:, ::1] D, const ITYPE_t n_jobs) nogil except -1:
        cdef:
            ITYPE_t n_samples = Y.shape[0]
            ITYPE_t i, this_n, end, start
            ITYPE_t i1, i2, num_threads
            ITYPE_t chunk_size = Y.shape[0] // n_jobs
            DTYPE_t dist
        start = 0
        num_threads = n_jobs
        if chunk_size == 0:
            chunk_size = n_samples
            num_threads = 1
        for i in prange(0, n_samples, chunk_size, num_threads = n_jobs, nogil=True):
            start = i
            end = min(i + chunk_size, n_samples)
            for i1 in range(X.shape[0]):
                for i2 in range(end-start):
                    dist = self.dist(&X[i1, 0], &Y[start+i2, 0], X.shape[1], Y.shape[1])
                    D[i1, start+i2] = dist
        return 0


    def pairwise(self, X, Y, n_jobs):
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] Xarr
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] Yarr
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] Darr

        Xarr = np.asarray(X, dtype=DTYPE, order='C')
        Yarr = np.asarray(Y, dtype=DTYPE, order='C')
        Darr = np.zeros((Xarr.shape[0], Yarr.shape[0]),
                        dtype=DTYPE, order='C')
        self.parallel_cdist1(Xarr, Yarr, Darr, n_jobs)
        return Darr

    def coocurences(self, X, Y, eps, n_jobs):
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] Xarr
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] Yarr
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] Larr

        cdef ITYPE_t i1, i2

        Xarr = np.asarray(X, dtype=DTYPE, order='C')
        Yarr = np.asarray(Y, dtype=DTYPE, order='C')
        Larr = np.zeros(Xarr.shape[0],
                        dtype=DTYPE, order='C')
        self.parallel_cdist(Xarr, Yarr, Larr, eps, n_jobs)
    
        return Larr

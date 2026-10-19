"""Pytest configuration: pin BLAS to one thread before numpy is imported.

The solvers work on small dense matrices, where multithreaded BLAS is far slower than one thread.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

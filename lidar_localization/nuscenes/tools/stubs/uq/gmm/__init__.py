def __getattr__(n):
    import numba
    return numba._T()

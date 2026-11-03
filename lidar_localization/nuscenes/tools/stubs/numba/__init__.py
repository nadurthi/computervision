"""No-op stand-in for numba so the original pure-Python bodies run unchanged (slowly)."""
class _T:
    def __getitem__(self, k): return self
    def __call__(self, *a, **k): return self
    def __getattr__(self, n): return _T()
def _deco(*a, **k):
    if len(a) == 1 and callable(a[0]) and not isinstance(a[0], _T) and not k: return a[0]
    return lambda f: f
njit = jit = vectorize = guvectorize = _deco
prange = range
types = _T()
float64 = float32 = int64 = int32 = double = uintc = boolean = _T()
cuda = _T()
def __getattr__(n): return _T()

"""TensorRT engines: build from ONNX with a pinned timing cache, and run them on torch tensors.

Determinism. The builder times candidate kernels ("tactics") and keeps the fastest; when two are
close, system noise decides, and different kernels accumulate floating point in different
orders, so two machines can build engines that give slightly different numbers. One touch more
or less changes every later skill label in a possession. The fix from the blog: build once,
save the timing cache, commit it next to the ONNX file, and build every later engine from that
cache with ignore_mismatch=False. Engines are also tied to a GPU model + TensorRT version, so the
cache key is recorded in a small JSON next to the engine.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_engine(onnx_path: str | Path, engine_path: str | Path, cache_path: str | Path,
                 fp16: bool = True, min_batch: int = 1, opt_batch: int = 16, max_batch: int = 32,
                 workspace_gb: float = 4.0, strict_cache: bool = False) -> Path:
    """Build (or reuse) an engine. Creates the timing cache on first build, reuses it afterwards.

    strict_cache=True refuses to build without an existing cache file: use it on every machine
    except the one that creates the cache, so a missing cache fails loudly instead of silently
    re-timing tactics.
    """
    import tensorrt as trt

    onnx_path, engine_path, cache_path = Path(onnx_path), Path(engine_path), Path(cache_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)          # explicit batch is the only mode in TensorRT 10
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # dynamic batch, if the ONNX graph allows it
    inp = network.get_input(0)
    shape = list(inp.shape)
    if shape[0] == -1:
        prof = builder.create_optimization_profile()
        rest = shape[1:]
        prof.set_shape(inp.name, [min_batch] + rest, [opt_batch] + rest, [max_batch] + rest)
        config.add_optimization_profile(prof)

    if cache_path.exists():
        cache = config.create_timing_cache(cache_path.read_bytes())
    elif strict_cache:
        raise FileNotFoundError(f"timing cache {cache_path} missing and strict_cache=True")
    else:
        cache = config.create_timing_cache(b"")
    config.set_timing_cache(cache, ignore_mismatch=False)

    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(blob))
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(memoryview(config.get_timing_cache().serialize()))

    import torch

    meta = {"onnx_sha": _sha(onnx_path), "cache_sha": _sha(cache_path), "engine_sha": _sha(engine_path),
            "tensorrt": trt.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "fp16": fp16}
    engine_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return engine_path


class TRTModule:
    """Minimal runner: torch CUDA tensors in, torch CUDA tensors out, on the current stream."""

    def __init__(self, engine_path: str | Path):
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self._dtype = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                       trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64,
                       trt.DataType.BOOL: torch.bool}
        b = self.engine.get_tensor_shape(self.inputs[0])[0]
        self.fixed_batch = b if b > 0 else None

    def _run(self, *xs):
        torch = self.torch
        for name, x in zip(self.inputs, xs):
            x = x.contiguous().to(self._dtype[self.engine.get_tensor_dtype(name)])
            self.ctx.set_input_shape(name, tuple(x.shape))
            self.ctx.set_tensor_address(name, x.data_ptr())
        outs = []
        for name in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(name))
            o = torch.empty(shape, dtype=self._dtype[self.engine.get_tensor_dtype(name)], device="cuda")
            self.ctx.set_tensor_address(name, o.data_ptr())
            outs.append(o)
        stream = torch.cuda.current_stream()
        if not self.ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return outs

    def __call__(self, *xs):
        n = xs[0].shape[0]
        if self.fixed_batch is None or self.fixed_batch == n:
            outs = self._run(*xs)
        else:   # batch-1 engine: loop
            parts = [self._run(*(x[i:i + 1] for x in xs)) for i in range(n)]
            outs = [self.torch.cat([p[k] for p in parts]) for k in range(len(self.outputs))]
        return outs if len(outs) > 1 else outs[0]


def events_fingerprint(dvw_path: str | Path) -> str:
    """Hash of the scout section only (header carries timestamps), for cross-machine checks."""
    text = Path(dvw_path).read_text(encoding="utf-8")
    scout = text[text.index("[3SCOUT]"):]
    return hashlib.sha256(scout.encode()).hexdigest()


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))

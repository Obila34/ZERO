"""TEN VAD via its official WebAssembly build, hosted in-process by wasmtime.

Why: TEN VAD ships native libraries only for x64-Linux, Android, Windows, iOS
and macOS — there is NO glibc-ARM64 build, so the pip package raises
"Unsupported platform: Linux aarch64" on the Raspberry Pi, and its feature-
extraction core is closed (not rebuildable for a new platform). The one
platform-independent artifact upstream ships is ``lib/Web/ten_vad.wasm`` — the
same model + feature extraction compiled to WebAssembly. We run THAT through
wasmtime (which has an ARM64 build), so the Pi runs the real TEN VAD, byte-for-
byte identical to x86, at ~1.1 ms per 16 ms frame (~7% of one core).

The .wasm is an Emscripten MODULARIZE build: it imports 6 host functions
(module "a") and exports memory + malloc/free/ten_vad_* under minified names.
We stub the 6 imports and drive the documented C API over linear memory. The
export/import mapping is read from the shipped ``ten_vad.js`` glue.

Validated frame-for-frame against the native x86 library
(scripts/validate_ten_wasm.py) — identical from a common start state.
"""
from __future__ import annotations

import ctypes
import struct
from pathlib import Path

import numpy as np

from zero.utils.logging import get_logger

log = get_logger("vad.ten_wasm")

# Export names from ten_vad.js: g=memory i=malloc j=free k=create l=process
# m=destroy h=__wasm_call_ctors. Imports a.a..a.f map to the glue's `ea` object.
_EXP = dict(mem="g", malloc="i", free="j", create="k", process="l",
            destroy="m", ctors="h")


class TenVadWasm:
    """Frame-level TEN VAD. ``process(frame)`` -> (probability, is_voice) for a
    ``hop_size``-sample int16 frame. Stateful (recurrent): feed frames in order;
    ``reset()`` clears the internal state for a new stream."""

    def __init__(self, wasm_path: str, hop_size: int = 256,
                 threshold: float = 0.5):
        import wasmtime

        if not wasm_path or not Path(wasm_path).exists():
            raise FileNotFoundError(f"ten_vad.wasm missing: {wasm_path}")
        self._wasmtime = wasmtime
        self.hop_size = int(hop_size)
        self.threshold = float(threshold)

        self._store = wasmtime.Store()
        self._module = wasmtime.Module(self._store.engine,
                                       Path(wasm_path).read_bytes())
        self._mem = None  # set after instantiate; import stubs close over self

        imports = self._build_imports()
        self._inst = wasmtime.Instance(self._store, self._module, imports)
        exp = self._inst.exports(self._store)
        self._mem = exp[_EXP["mem"]]
        self._malloc = exp[_EXP["malloc"]]
        self._free = exp[_EXP["free"]]
        self._create = exp[_EXP["create"]]
        self._process = exp[_EXP["process"]]
        self._destroy = exp[_EXP["destroy"]]
        exp[_EXP["ctors"]](self._store)  # __wasm_call_ctors

        # Persistent scratch buffers (audio in, prob/flag out) + the VAD handle.
        self._audio_ptr = self._malloc(self._store, self.hop_size * 2)
        self._prob_ptr = self._malloc(self._store, 4)
        self._flag_ptr = self._malloc(self._store, 4)
        self._handle_ptr = self._malloc(self._store, 4)
        self._handle = 0
        self._create_handle()
        log.info("TEN VAD (wasm) loaded (%s, hop=%d, threshold=%.2f)",
                 wasm_path, self.hop_size, self.threshold)

    # -- host imports (a.a..a.f), matching ten_vad.js `ea`. Types come from the
    #    module, so returns must match exactly (a.f = emscripten_memcpy_big = void).
    def _build_imports(self):
        wt = self._wasmtime

        def _base():
            return ctypes.addressof(ctypes.cast(
                self._mem.data_ptr(self._store),
                ctypes.POINTER(ctypes.c_ubyte)).contents)

        def abort(*_):
            raise wt.Trap("ten_vad wasm abort")

        def fd_seek(*_):
            return 70

        def resize_heap(requested, *_):
            requested &= 0xFFFFFFFF
            cur = self._mem.size(self._store) * 65536
            if requested <= cur:
                return 1
            try:
                self._mem.grow(self._store,
                               (requested - cur + 65535) // 65536)
                return 1
            except Exception:
                return 0

        def fd_write(fd, iov, iovcnt, pnum, *_):
            total = 0
            for i in range(iovcnt):
                total += struct.unpack_from(
                    "<I", self._read(iov + i * 8 + 4, 4))[0]
            self._write(pnum, struct.pack("<I", total))
            return 0

        def fd_close(*_):
            return 52

        def memcpy_big(dest, src, num, *_):
            b = _base()
            ctypes.memmove(b + dest, b + src, num)
            return None

        fns = [abort, fd_seek, resize_heap, fd_write, fd_close, memcpy_big]
        return [wt.Func(self._store, imp.type, fn)
                for imp, fn in zip(self._module.imports, fns)]

    # -- linear-memory helpers --
    def _addr(self, off: int) -> int:
        base = ctypes.addressof(ctypes.cast(
            self._mem.data_ptr(self._store),
            ctypes.POINTER(ctypes.c_ubyte)).contents)
        return base + off

    def _read(self, off: int, n: int) -> bytes:
        return bytes((ctypes.c_ubyte * n).from_address(self._addr(off)))

    def _write(self, off: int, data: bytes) -> None:
        ctypes.memmove(self._addr(off), data, len(data))

    def _create_handle(self) -> None:
        rc = self._create(self._store, self._handle_ptr, self.hop_size,
                          self.threshold)
        self._handle = struct.unpack("<I", self._read(self._handle_ptr, 4))[0]
        if rc != 0 or self._handle == 0:
            raise RuntimeError(f"ten_vad_create failed (rc={rc})")

    def reset(self) -> None:
        """Clear recurrent state — start of a new utterance/stream."""
        try:
            if self._handle:
                self._destroy(self._store, self._handle_ptr)
        except Exception as e:
            log.debug("ten_vad destroy failed: %s", e)
        self._create_handle()

    def process(self, frame_i16: np.ndarray) -> tuple[float, int]:
        """One ``hop_size``-sample int16 frame -> (probability, is_voice 0/1).
        Frames shorter/longer than hop_size are zero-padded/truncated."""
        f = np.asarray(frame_i16).astype("<i2", copy=False).reshape(-1)
        if f.size != self.hop_size:
            g = np.zeros(self.hop_size, dtype="<i2")
            g[:min(f.size, self.hop_size)] = f[:self.hop_size]
            f = g
        self._write(self._audio_ptr, f.tobytes())
        self._process(self._store, self._handle, self._audio_ptr,
                      self.hop_size, self._prob_ptr, self._flag_ptr)
        prob = struct.unpack("<f", self._read(self._prob_ptr, 4))[0]
        flag = struct.unpack("<i", self._read(self._flag_ptr, 4))[0]
        return prob, flag

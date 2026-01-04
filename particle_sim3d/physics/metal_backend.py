from __future__ import annotations

import struct
import sys
from typing import Any

import numpy as np


class MetalNBody:
    def __init__(self, *, tile_size: int = 256) -> None:
        try:
            from Metal import (  # type: ignore
                MTLCompileOptions,
                MTLCreateSystemDefaultDevice,
                MTLResourceStorageModeShared,
                MTLSizeMake,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Missing Metal bindings: install pyobjc-framework-Metal.") from exc

        self._mtl = {
            "MTLResourceStorageModeShared": MTLResourceStorageModeShared,
            "MTLSizeMake": MTLSizeMake,
        }

        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:  # pragma: no cover - depends on platform
            raise RuntimeError("Metal device not available.")
        self.queue = self.device.newCommandQueue()
        if self.queue is None:  # pragma: no cover - depends on platform
            raise RuntimeError("Metal command queue unavailable.")

        self.tile_size = int(tile_size)
        if self.tile_size <= 0:
            raise ValueError("tile_size must be > 0")

        source = self._build_shader_source(self.tile_size)
        options = MTLCompileOptions.new()
        library, error = self.device.newLibraryWithSource_options_error_(source, options, None)
        if library is None:  # pragma: no cover - compilation failure
            message = str(error) if error is not None else "unknown compile error"
            raise RuntimeError(f"Metal shader compile failed: {message}")

        function = library.newFunctionWithName_("accel_kernel")
        if function is None:  # pragma: no cover - compilation failure
            raise RuntimeError("Metal shader missing accel_kernel.")

        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None:  # pragma: no cover - pipeline failure
            message = str(error) if error is not None else "unknown pipeline error"
            raise RuntimeError(f"Metal pipeline creation failed: {message}")

        max_threads = int(pipeline.maxTotalThreadsPerThreadgroup())
        if self.tile_size > max_threads:
            raise RuntimeError(
                f"tile_size {self.tile_size} exceeds max threads per threadgroup ({max_threads})."
            )

        self.pipeline = pipeline
        self._buffers: dict[str, Any] = {}

    def compute_accel(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        signs: np.ndarray,
        *,
        g: float,
        eps2: float,
        debug: bool = False,
    ) -> np.ndarray:
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        chg = np.ascontiguousarray(charges, dtype=np.float32)
        sgn = np.ascontiguousarray(signs, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] not in (3, 4):
            raise ValueError("positions must be Nx3 or Nx4")
        n = int(pos.shape[0])
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if chg.shape[0] != n or sgn.shape[0] != n:
            raise ValueError("charges/signs length mismatch")

        if pos.shape[1] == 3:
            pos4 = np.empty((n, 4), dtype=np.float32)
            pos4[:, :3] = pos
            pos4[:, 3] = 0.0
        elif pos.shape[1] == 4:
            pos4 = pos
        else:
            raise ValueError("positions must be Nx3 or Nx4")

        buf_pos = self._ensure_buffer("pos", pos4.nbytes)
        buf_chg = self._ensure_buffer("chg", chg.nbytes)
        buf_sgn = self._ensure_buffer("sgn", sgn.nbytes)
        buf_out = self._ensure_buffer("out", n * 4 * 4)

        self._memcpy(buf_pos, pos4)
        self._memcpy(buf_chg, chg)
        self._memcpy(buf_sgn, sgn)

        debug_flag = 1.0 if debug else 0.0
        params = struct.pack("Ifff", n, float(g), float(eps2), float(debug_flag))

        cmd_buffer = self.queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline)
        encoder.setBuffer_offset_atIndex_(buf_pos, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_chg, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_sgn, 0, 2)
        encoder.setBuffer_offset_atIndex_(buf_out, 0, 3)
        if hasattr(encoder, "setBytes_length_atIndex_"):
            encoder.setBytes_length_atIndex_(params, len(params), 4)
        else:
            buf_params = self._ensure_buffer("params", len(params))
            self._memcpy(buf_params, params)
            encoder.setBuffer_offset_atIndex_(buf_params, 0, 4)

        if debug:
            nan_out = np.full((n, 4), np.nan, dtype=np.float32)
            self._memcpy(buf_out, nan_out)

        threads = self._mtl["MTLSizeMake"](self.tile_size, 1, 1)
        group_count = max(1, (n + self.tile_size - 1) // self.tile_size)
        groups = self._mtl["MTLSizeMake"](group_count, 1, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(groups, threads)
        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        accel4 = self._buffer_to_numpy(buf_out, n)
        if debug:
            nan_mask = np.isnan(accel4[:, :3]).any(axis=1)
            if np.any(nan_mask):
                count = int(np.sum(nan_mask))
                print(f"[metal][debug] unwritten outputs (NaN): {count}", file=sys.stderr)
            if n >= 1:
                w0 = float(accel4[0, 3])
                print(f"[metal][debug] param marker g: {w0:.6g}", file=sys.stderr)
            if n >= 2:
                w1 = float(accel4[1, 3])
                print(f"[metal][debug] param marker eps2: {w1:.6g}", file=sys.stderr)
            if n >= 3:
                w2 = float(accel4[2, 3])
                print(f"[metal][debug] param marker n: {w2:.6g}", file=sys.stderr)

            try:
                pos_back = self._buffer_to_numpy(buf_pos, n)
                pos_diff = np.max(np.abs(pos_back - pos4))
                print(f"[metal][debug] pos max abs diff: {pos_diff:.6g}", file=sys.stderr)
            except Exception as exc:
                print(f"[metal][debug] pos readback failed: {exc}", file=sys.stderr)

            try:
                chg_view = self._buffer_view(buf_chg, n * 4)
                chg_back = np.frombuffer(chg_view[: n * 4], dtype=np.float32, count=n)
                chg_diff = float(np.max(np.abs(chg_back - chg)))
                print(f"[metal][debug] charge max abs diff: {chg_diff:.6g}", file=sys.stderr)
            except Exception as exc:
                print(f"[metal][debug] charge readback failed: {exc}", file=sys.stderr)

            try:
                sgn_view = self._buffer_view(buf_sgn, n * 4)
                sgn_back = np.frombuffer(sgn_view[: n * 4], dtype=np.float32, count=n)
                sgn_diff = float(np.max(np.abs(sgn_back - sgn)))
                print(f"[metal][debug] sign max abs diff: {sgn_diff:.6g}", file=sys.stderr)
            except Exception as exc:
                print(f"[metal][debug] sign readback failed: {exc}", file=sys.stderr)

        return accel4[:, :3].copy()

    def _ensure_buffer(self, key: str, length: int) -> Any:
        buf = self._buffers.get(key)
        if buf is None or int(buf.length()) < length:
            buf = self.device.newBufferWithLength_options_(
                length, self._mtl["MTLResourceStorageModeShared"]
            )
            self._buffers[key] = buf
        return buf

    def _buffer_view(self, buf: Any, length: int) -> memoryview:
        contents = buf.contents()
        if hasattr(contents, "as_buffer"):
            try:
                view = memoryview(contents.as_buffer(length)).cast("B")
                if view.nbytes >= length:
                    return view
            except Exception:
                pass
        try:
            view = memoryview(contents).cast("B")
            if view.nbytes >= length:
                return view
        except Exception:
            pass
        raise RuntimeError(f"Unable to access MTLBuffer contents (type={type(contents)})")

    def _memcpy(self, buf: Any, data: Any) -> None:
        if isinstance(data, np.ndarray):
            src = memoryview(data).cast("B")
            view = self._buffer_view(buf, src.nbytes)
            view[: src.nbytes] = src
            return
        if isinstance(data, (bytes, bytearray)):
            view = self._buffer_view(buf, len(data))
            view[: len(data)] = data
            return
        raise TypeError(f"Unsupported buffer source: {type(data)}")

    def _buffer_to_numpy(self, buf: Any, n: int) -> np.ndarray:
        count = n * 4
        view = self._buffer_view(buf, count * 4)
        arr = np.frombuffer(view[: count * 4], dtype=np.float32, count=count)
        return arr.reshape((n, 4))

    @staticmethod
    def _build_shader_source(tile_size: int) -> str:
        return f"""
#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE {tile_size}

struct Params {{
    uint n;
    float g;
    float eps2;
    float debug;
}};

kernel void accel_kernel(
    device const float4* positions [[buffer(0)]],
    device const float* charges [[buffer(1)]],
    device const float* signs [[buffer(2)]],
    device float4* out_accel [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tpg [[threadgroup_position_in_grid]]
) {{
    uint tid = tpg.x * TILE_SIZE + lid;
    bool active = tid < params.n;
    float3 pi = active ? positions[tid].xyz : float3(0.0f);
    float si = active ? signs[tid] : 0.0f;
    float3 acc = float3(0.0f);

    threadgroup float4 tpos[TILE_SIZE];
    threadgroup float tchg[TILE_SIZE];

    for (uint base = 0; base < params.n; base += TILE_SIZE) {{
        uint idx = base + lid;
        if (idx < params.n) {{
            tpos[lid] = positions[idx];
            tchg[lid] = charges[idx];
        }} else {{
            tpos[lid] = float4(0.0f);
            tchg[lid] = 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint count = min((uint)TILE_SIZE, params.n - base);
        if (active) {{
            for (uint j = 0; j < count; ++j) {{
                uint jj = base + j;
                if (jj == tid) {{
                    continue;
                }}
                float3 d = tpos[j].xyz - pi;
                float r2 = dot(d, d) + params.eps2;
                if (!isfinite(r2)) {{
                    continue;
                }}
                r2 = max(r2, 1.0e-12f);
                float inv_r = rsqrt(r2);
                if (!isfinite(inv_r)) {{
                    continue;
                }}
                float inv_r3 = inv_r * inv_r * inv_r;
                if (!isfinite(inv_r3)) {{
                    continue;
                }}
                float f = params.g * si * tchg[j] * inv_r3;
                if (!isfinite(f)) {{
                    continue;
                }}
                acc += d * f;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (active) {{
        float w = 0.0f;
        if (params.debug > 0.5f) {{
            if (tid == 0) {{
                w = params.g;
            }} else if (tid == 1) {{
                w = params.eps2;
            }} else if (tid == 2) {{
                w = float(params.n);
            }}
        }}
        out_accel[tid] = float4(acc, w);
    }}
}}
"""

"""
TVM FFI Bindings for GDN Prefill CUDA Kernel.

Signature (DPS style):
  run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)
"""

import ctypes
import os
import torch
from tvm.ffi import register_func

_lib_path = os.path.join(os.path.dirname(__file__), "kernel.so")
_lib = ctypes.CDLL(_lib_path)

_lib.launch_gdn_prefill.restype = None
_lib.launch_gdn_prefill.argtypes = [
    ctypes.c_void_p,  # q
    ctypes.c_void_p,  # k
    ctypes.c_void_p,  # v
    ctypes.c_void_p,  # state
    ctypes.c_void_p,  # A_log
    ctypes.c_void_p,  # a
    ctypes.c_void_p,  # dt_bias
    ctypes.c_void_p,  # b_gate
    ctypes.c_void_p,  # cu_seqlens
    ctypes.c_float,   # scale
    ctypes.c_void_p,  # output
    ctypes.c_void_p,  # new_state
    ctypes.c_int,     # num_seqs
]


@register_func("flashinfer.kernel")
def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    num_seqs = cu_seqlens.shape[0] - 1

    _lib.launch_gdn_prefill(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(state.data_ptr()),
        ctypes.c_void_p(A_log.data_ptr()),
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(dt_bias.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(cu_seqlens.data_ptr()),
        ctypes.c_float(scale),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(new_state.data_ptr()),
        ctypes.c_int(num_seqs),
    )

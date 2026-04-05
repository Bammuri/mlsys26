"""
TVM FFI Bindings for GDN Decode CUDA Kernel.

Signature (DPS style):
  run(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state)

Inputs:
  q[B,1,4,128] bf16, k[B,1,4,128] bf16, v[B,1,8,128] bf16,
  state[B,8,128,128] f32, A_log[8] f32, a[B,1,8] bf16,
  dt_bias[8] f32, b[B,1,8] bf16, scale f32

Outputs (pre-allocated):
  output[B,1,8,128] bf16, new_state[B,8,128,128] f32
"""

import ctypes
import os
import torch
from tvm.ffi import register_func

# Load the compiled shared library
_lib_path = os.path.join(os.path.dirname(__file__), "kernel.so")
_lib = ctypes.CDLL(_lib_path)

# Set up function signature
_lib.launch_gdn_decode.restype = None
_lib.launch_gdn_decode.argtypes = [
    ctypes.c_void_p,  # q
    ctypes.c_void_p,  # k
    ctypes.c_void_p,  # v
    ctypes.c_void_p,  # state
    ctypes.c_void_p,  # A_log
    ctypes.c_void_p,  # a
    ctypes.c_void_p,  # dt_bias
    ctypes.c_void_p,  # b_gate
    ctypes.c_float,   # scale
    ctypes.c_void_p,  # output
    ctypes.c_void_p,  # new_state
    ctypes.c_int,     # batch_size
]


@register_func("flashinfer.kernel")
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    batch_size = q.shape[0]

    _lib.launch_gdn_decode(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(state.data_ptr()),
        ctypes.c_void_p(A_log.data_ptr()),
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(dt_bias.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_float(scale),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(new_state.data_ptr()),
        ctypes.c_int(batch_size),
    )

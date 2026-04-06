"""
Modal-based NCU runner for GDN prefill.

This script shifts NCU execution to Modal GPU containers instead of the local
machine. It supports a lightweight environment probe and a full `ncu` capture
path that wraps `scripts/profile_ncu_prefill.py`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import modal

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LOCAL_PROJECT_ROOT))

TRACE_VOLUME_NAME = "mlsys26-contest"
ARTIFACT_VOLUME_NAME = "mlsys26-ncu"
TRACE_SET_PATH = "/data/data/mlsys26-contest"
ARTIFACT_PATH = "/artifacts"
REMOTE_PROJECT_ROOT = "/workspace"

app = modal.App("flashinfer-bench-ncu")

trace_volume = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)

BASE_PATH = "/opt/nvidia/nsight-compute:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "PATH": BASE_PATH,
            "FIB_DATASET_PATH": TRACE_SET_PATH,
        }
    )
    .pip_install("flashinfer-bench", "flashinfer-python", "torch", "triton", "numpy")
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "scripts"),
        remote_path=f"{REMOTE_PROJECT_ROOT}/scripts",
        copy=True,
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "solution"),
        remote_path=f"{REMOTE_PROJECT_ROOT}/solution",
        copy=True,
    )
    .add_local_file(
        str(LOCAL_PROJECT_ROOT / "config.toml"),
        remote_path=f"{REMOTE_PROJECT_ROOT}/config.toml",
        copy=True,
    )
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _run_shell(cmd: str) -> dict:
    proc = subprocess.run(
        ["/bin/bash", "-lc", cmd],
        capture_output=True,
        text=True,
        cwd=REMOTE_PROJECT_ROOT,
        env=os.environ.copy(),
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={"/data": trace_volume, ARTIFACT_PATH: artifact_volume},
)
def check_ncu_env() -> dict:
    cmd = r"""
set -euo pipefail
echo "timestamp=$(date -Iseconds)"
echo "pwd=$(pwd)"
echo "python=$(python -V 2>&1)"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "PATH=$PATH"
echo "--- which ---"
which ncu || true
which nvcc || true
echo "--- versions ---"
ncu --version || true
nvcc --version | head -n 5 || true
echo "--- gpu ---"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("torch_cuda", torch.version.cuda)
PY
"""
    return _run_shell(cmd)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=7200,
    volumes={"/data": trace_volume, ARTIFACT_PATH: artifact_volume},
)
def capture_ncu(
    workload_uuid: str,
    ncu_set: str = "detailed",
    report_name: str = "",
    kernel_name: str = "gdn_prefill_kernel",
    launch_skip: int = 3,
    launch_count: int = 1,
) -> dict:
    safe_kernel = kernel_name.replace(":", "_").replace("(", "_").replace(")", "_")
    report_name = report_name or f"{safe_kernel}-{workload_uuid[:8]}-{_timestamp()}"
    report_base = Path(ARTIFACT_PATH) / report_name
    cmd = f"""
set -euo pipefail
mkdir -p {shlex.quote(ARTIFACT_PATH)}
echo "report_base={report_base}"
which ncu
ncu --version
ncu --set {shlex.quote(ncu_set)} \\
  --kernel-name {shlex.quote(kernel_name)} \\
  --launch-skip {int(launch_skip)} --launch-count {int(launch_count)} \\
  -o {shlex.quote(str(report_base))} \\
  python scripts/profile_ncu_prefill.py --workload-uuid {shlex.quote(workload_uuid)}
ls -lah {shlex.quote(ARTIFACT_PATH)}
"""
    result = _run_shell(cmd)

    try:
        artifact_volume.commit()
    except Exception:
        pass

    return {
        **result,
        "report_base": str(report_base),
        "artifacts_path": ARTIFACT_PATH,
        "volume_name": ARTIFACT_VOLUME_NAME,
    }


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={"/data": trace_volume, ARTIFACT_PATH: artifact_volume},
)
def inspect_ncu_report(
    report_name: str,
    page: str = "details",
    csv: bool = False,
) -> dict:
    report_path = Path(ARTIFACT_PATH) / (
        report_name if report_name.endswith(".ncu-rep") else f"{report_name}.ncu-rep"
    )
    csv_flag = "--csv" if csv else ""
    cmd = f"""
set -euo pipefail
test -f {shlex.quote(str(report_path))}
ncu --import {shlex.quote(str(report_path))} --page {shlex.quote(page)} {csv_flag}
"""
    return {
        **_run_shell(cmd),
        "report_path": str(report_path),
    }


@app.local_entrypoint()
def main(
    check_env: bool = False,
    workload_uuid: str = "",
    inspect_report: str = "",
    page: str = "details",
    csv: bool = False,
    ncu_set: str = "detailed",
    report_name: str = "",
    kernel_name: str = "gdn_prefill_kernel",
    launch_skip: int = 3,
    launch_count: int = 1,
):
    if check_env:
        result = check_ncu_env.remote()
        print(result["stdout"], end="")
        if result["stderr"]:
            print(result["stderr"], file=sys.stderr, end="")
        if result["returncode"] != 0:
            raise SystemExit(result["returncode"])
        return

    if inspect_report:
        result = inspect_ncu_report.remote(report_name=inspect_report, page=page, csv=csv)
        print(result["stdout"], end="")
        if result["stderr"]:
            print(result["stderr"], file=sys.stderr, end="")
        if result["returncode"] != 0:
            raise SystemExit(result["returncode"])
        return

    if not workload_uuid:
        raise SystemExit("Pass --check-env, --inspect-report <NAME>, or provide --workload-uuid <UUID>.")

    result = capture_ncu.remote(
        workload_uuid=workload_uuid,
        ncu_set=ncu_set,
        report_name=report_name,
        kernel_name=kernel_name,
        launch_skip=launch_skip,
        launch_count=launch_count,
    )
    print(result["stdout"], end="")
    if result["stderr"]:
        print(result["stderr"], file=sys.stderr, end="")
    print(
        f"\n[modal-ncu] report_base={result['report_base']} "
        f"(volume={result['volume_name']})"
    )
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])

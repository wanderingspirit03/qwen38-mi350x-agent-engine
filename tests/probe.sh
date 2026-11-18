#!/bin/bash
cd "$(dirname "$0")/.."
sudo -n /opt/ij-amd/amd/controller/box/ij-docker build . 2>&1 | tail -3
sudo -n /opt/ij-amd/amd/controller/box/ij-docker run --rm ij-builder/engine:latest bash -c 'python3 -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(), torch.version.hip)"; pip list 2>/dev/null | grep -i -E "torch|triton|safetensors|aiter|flash|numpy|transformers"; which hipcc; ls /opt/rocm/lib | grep -i -E "hipblaslt|rocblas" | head; nproc; free -g'

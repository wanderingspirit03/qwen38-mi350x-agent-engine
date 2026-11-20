# Qwen3.8-27B Inference Engine for AMD Instinct MI350X

A standalone inference engine for `Qwen/Qwen3.8-27B` on a single AMD Instinct MI350X (CDNA 4, 288 GB HBM), written autonomously by coding agents (`claude-opus-5-5` lead with `claude-opus-5-5` and `claude-sonnet-5-5` subagents) inside `prime-agent` over 32.3 hours with no human code edits.

The engine was built from a starting repository with no engine code (`7305ab1`) and graded by an isolated judge and harness in [`inference-judge/research/amd`](https://github.com/wanderingspirit03/inference-judge/tree/main/research/amd). The frozen evaluated build is commit `657edeed1d3667293ee1ed91656b3012293b2cb6` (tree-identical to the agents' final commit `ec22955` and tagged `evaluated-657edee`). The only commit after `ec22955` adds this README.

## Headline results

Evaluated after the run on two unseen 12-prompt coding-agent sets (6,324 to 12,017 input tokens, 128-token output cap, one request at a time, same MI350X droplet) against AMD's tuned `proaditya/ATOM` megakernel path (`qwen38-megakernels`, FlyDSL `0.4.0.dev973`, DFlash2) and the BF16 ATOM reference:

| Unseen set | Engine | Whole-request (tok/s) | vs. AMD | vs. BF16 | Mean gap (nats) | Bar |
|---|---|---:|---:|---:|---:|---|
| Set 0 | BF16 reference | 58.11 | 0.32x | 1.00x | 0.0000 | PASS |
| Set 0 | AMD tuned (`proaditya/ATOM`) | 182.67 | 1.00x | 3.14x | 0.0230 | PASS |
| Set 0 | Agent engine (`657edee`) | 272.39 | 1.49x | 4.69x | 0.0492 | PASS |
| Set 1 | BF16 reference | 58.06 | 0.33x | 1.00x | 0.0000 | PASS |
| Set 1 | AMD tuned (`proaditya/ATOM`) | 177.96 | 1.00x | 3.07x | 0.0247 | PASS |
| Set 1 | Agent engine (`657edee`) | 272.79 | 1.53x | 4.70x | 0.0514 | PASS |

Across all 24 matched unseen prompts, the agent engine was faster than AMD's tuned path on every prompt, with a paired geometric mean speedup of 1.51x (95% bootstrap CI 1.47x to 1.55x across prompts) and 4.77x over BF16.

Two qualifications matter when reading those numbers:

- **Most of the gain is prefill.** On long coding-agent prompts with 128-token replies, prompt processing accounts for roughly four fifths of the time saved (72% to 95% across AMD's diagnostic decode range). Under exact verification on the development pool, the two engines decode at roughly the same pace.
- **Lossy draft acceptance trades fidelity for extra decode speed.** Both engines cleared the same fixed quality bar (top-5 share >= 0.98, mean log-probability gap <= 0.08 nats, 64-task paired LB95 >= -0.15). With the default `ENG_LOSSY_TAU=0.30`, the agent engine sat roughly twice as far from the BF16 reference as AMD's path and lost 6 of 128 paired tasks on its unseen draws. Pass `ENG_LOSSY_TAU=0` to disable lossy acceptance and run exact verification against the quantized target.

## How it works

The serving path (`eng/`, `Dockerfile`, `serve`) spans 45 files and 7,909 lines of Python, Triton, and HIP C++ without importing or calling ATOM, vLLM, or SGLang:

- **Chunked FP8 and mixed-precision prefill (`eng/prefill.py`, `eng/kernels/f8mm.py`, `eng/kernels/attn2.py`, `eng/csrc/gdn_wy.hip`, `gdn_scan.hip`).** Up to 8,192 tokens per chunk, custom XCD-aware Triton FP8 GEMMs, Stream-K mixed INT8 QK / FP8 PV causal attention on long chunks, and Triton/HIP chunked Gated DeltaNet updates.
- **Hybrid GPU prefix cache (`eng/cache.py`).** 80 GiB default GPU budget with LRU eviction, caching both GQA KV tensors and GDN recurrent/convolution states across requests.
- **Asymmetric INT4/INT8 MFMA decode with startup GPTQ (`eng/kernels/w4a8.py`, `eng/gptq.py`).** Custom integer MFMA decode kernels (group size 128, FP16 scale and zero point, INT8 activations and LM head) calibrated at startup over 9 public coding prompts (`eng/calib.npz`, ~5 min startup).
- **Fine-tuned native MTP drafter (`eng/mtp_delta.pt`, `eng/fast.py`).** Rank-64 LoRA adaptation of Qwen's bundled single-layer `mtp.layers.0` (14 MB merged delta, `K = 9`, 256 dynamic prompt-token vocab slots) running inside captured ROCm graphs with a fused HIP GDN verify kernel (`eng/csrc/small.hip` and `05e0dbe`).

## Running on an MI350X

Requirements: one AMD Instinct MI350X (`gfx950`), ROCm container runtime, `rocm/atom-dev` base image, and `Qwen/Qwen3.8-27B` weights on disk.

```bash
# Build the container (compiles eng/csrc/*.hip for gfx950)
docker build -t qwen38-agent-engine .

# Run the HTTP server on port 8000 (default tau = 0.30)
docker run --rm --shm-size 16g --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=private --network=host \
  -v /srv/models/Qwen3.8-27B:/models/Qwen3.8-27B:ro \
  qwen38-agent-engine python3 -m eng.main 8000

# To run exact verification against the quantized target (disable lossy acceptance)
docker run --rm --shm-size 16g --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=private --network=host \
  -e ENG_LOSSY_TAU=0 \
  -v /srv/models/Qwen3.8-27B:/models/Qwen3.8-27B:ro \
  qwen38-agent-engine python3 -m eng.main 8000
```

Startup runs GPTQ calibration and graph capture (~5 minutes on MI350X) before `GET /v1/models` returns 200. The server accepts `POST /v1/completions` with prompts supplied as token IDs and returns generated token IDs.

## Repository layout

```text
eng/                  Standalone serving runtime (Python, Triton, HIP C++)
  kernels/            Custom Triton prefill GEMM, mixed-precision attention, W4A8/W4 MFMA decode
  csrc/               Custom HIP C++ kernels (GDN WY/scan/verify, acceptance, streaming MTP GEMV)
  mtp_delta.pt        14 MB merged rank-64 LoRA delta for Qwen's mtp.layers.0
  calib.npz           9 public coding prompts used for startup GPTQ calibration
Dockerfile            Compiles HIP kernels for gfx950 on top of rocm/atom-dev
serve                 Entrypoint invoked by the inference-judge harness
tests/                Agent-written unit checks, microbenchmarks, and public prompt slices
AGENTS.md             Task brief given to the agents at the start of the run
```

## Key commits

| Commit | Milestone |
|---|---|
| `7305ab1` | Starting repository (brief and harness commands, no engine code) |
| `66c6f8c` | First passing submission (`#2`): FP8 linears and Triton GDN prefill |
| `518dfe4` | Custom INT4 decode kernels (`#3`) |
| `411116e` | W4A8 integer MFMA default (`#9`) |
| `a7c1063` | Mixed-precision prefill attention (`#13`) |
| `1475d28` | Startup GPTQ + rank-64 LoRA MTP adapter, exact verification (`#17`) |
| `f90aa7d` | Margin-based lossy acceptance `tau = 0.30` (`#18`) |
| `05e0dbe` | Fused single-kernel HIP GDN verify (`#19`) |
| `657edee` | Evaluated frozen build (tagged `evaluated-657edee`; tree-identical to `ec22955`) |

## Related repository

The isolated judge, container sandbox, single-GPU queue, cheat suite, and AMD baseline build scripts live in [`wanderingspirit03/inference-judge`](https://github.com/wanderingspirit03/inference-judge/tree/main/research/amd).

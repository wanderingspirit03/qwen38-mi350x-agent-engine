# mtpstream (sparse24): M=1 streaming GEMV for the MTP draft step — STOPPED by the lead

Branch `mtpstream` (worktree /home/agents/wt-ms2). Kernel `w4_gemv_kernel` in eng/kernels/_w4_src.py (same w4
packed format), `w4.gemv()`, `gemm()` dispatches to it at M=1 when ENG_GEMV=1 (default on that branch);
bench: `python3 -m eng.bench_gv` (random weights, rotated copies, HIP graphs; correctness vs fp32 reference).

Results (us, random weights), best new config vs old w4 kernel:
| | old | new (bf16 x) |
|---|---|---|
| fc 5120x10240 | 11.1 | 9.7 (1,4,2) |
| qkv 14336x5120 | 12.3 | 11.5 (2,4,2) |
| o 5120x6144 | 8.7 | 6.9 (1,4,2) |
| gate_up 34816x5120 | 26.0 | 25.7 (2,4,2) |
| down 5120x17408 | 16.3 | 18.5 |
| draft head 33024x5120 | 24.5 | 25.3 |
Errors identical to the old kernel (2-3e-3 rel vs fp32), argmax identical.
Lessons: copying ring registers (`f = ring[d]`) made the compiler drain vmcnt(0) every iteration; fully unrolled
loops hoist all LDS reads (190 VGPRs, occupancy 2); waves_per_eu(6) spills. The kernel seems bound by the bf16 MFMA
+ dequant work at M=1 (4 MFMA + 16 VALU per tile and 128-k block). An I8 variant (x int8 per block in the kernel,
2 i8 MFMAs, like the study) is written but untested (ENG_GEMV_I8). The study's gains were mostly an
uninitialized-buffer artifact (w4a8), so the port was stopped.

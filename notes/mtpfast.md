# mtpfast: making the MTP draft step faster (branch mtpfast, /home/agents/wt-mf)

All timings use random weights, or real weights where noted, in CUDA-graph replay.

## Baseline (eng/bench_mtp 9, real weights, ENG_GEMV=0)
- Chain of 8 steps: 120.3 us per step. One step replayed alone: 123.6 us.
- 11 kernels per step. The 5 small ones (mtp_in, attn_prep, decode split+merge, add_rmsnorm) take about 26 us. The 6 GEMMs take about 96 us.

## Windowed draft attention (sink 4 + last W keys): REJECTED
- Tested with ENG_MTP_WIN=W via eng.lossy_sweep 0.3 on tests/fprompts.jsonl, K=9.
- Tokens per round:

  | window | tokens/round |
  |---|---|
  | full | 7.76 |
  | W=16384 (same as full) | 7.71 |
  | W=1024 | 6.44 (-17%) |
  | W=256 | 5.24 |
  | W=64 | 5.20 |

- The draft needs the full context. The paper's result does not carry over to these long prompts.
- The fused kernel H.mtp_attn (attn_prep + attention + gate in one kernel, eng/test_mtp_attn.py) is correct but slow. It takes 48 us at W=252, because one WG per head walks the keys serially. It is kept only for the experiment (ENG_MTP_ATTN_HIP).
- What remains here: a single full-context split+merge kernel could save about 5 us, the launch and the merge pass.

## M=1 GEMV (from branch mtpstream), now on by default with tuned GV_CONFIGS
- Real chain: 115.0 us per step vs 120.3 (-5.3 us).
- The isolated benchmark (bench_gv) promised more: about 103 -> 88 us summed. In the chain, gate_up and down run slower than in isolation (27.9 / 24.2 us in the trace).
- I8 (x quantized to int8, i8 MFMA): no faster, except on down (13.8 vs 15.1). Its relative error is 2-4x higher. Left off (ENG_GEMV_I8=0).

## Where the GEMV time goes (eng/bench_gv_abl.py, eng/bench_rd.py)
- Plain read (rd_bench) runs at the same speed on random, zero and uninitialized buffers: 4.0 TB/s at 16 MB, 5.5 at 47 MB, 6.2 at 93 MB, 6.5 at 311 MB. So data content does not change read speed.
- Ablation, GEMV with its math removed:

  | shape | full | no math | loads only |
  |---|---|---|---|
  | gate_up | 24.7 us | 17.2-19.4 | 15.5 (= plain read) |
  | down | 16.0 | 12.3 | 10.3 |
  | qkv | 10.6 | 9.1 | 8.2 |

- So for gate_up, about 9 us is compute that does not overlap with the memory traffic.
- Grid balance is poor. With occupancy 4 there are 16 waves per CU, 256 CUs.
  - N=5120 shapes (down, o, fc): only 160 WGs, so 96 CUs sit idle.
  - qkv: 448 WGs, so CUs get 1 or 2 WGs each.
  - gate_up (2,4,2): 4352 waves on 4096 slots, so a 6% tail round.
- Next: stream-K or split-K. A persistent grid sized to the CUs with an fp32 atomic partial sum and a last-WG epilogue would fill every CU and smooth the tail. Expected: down 16 -> ~9, o 6.4 -> ~4, fc 8.2 -> ~5.5, qkv 11 -> ~7.5.

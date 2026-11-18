# This repository

This repository starts empty. It will hold an inference engine for one model
on one GPU. A separate program, the judge, measures the engine.

## The machine
- GPU: one AMD Instinct MI350X, 288 GB, architecture gfx950, as a virtual
  function inside a virtual machine. ROCm 10.0. 24 CPUs, 251 GB RAM.
- The model: Qwen3.8-27B. Weights are at `/srv/models/Qwen3.8-27B`,
  read-only. `/srv/models` also holds `Qwen3.8-27B-DFlash2`,
  the draft model AMD's tuned engine (below) uses.
- The GPU machine is rented at spot price and can be taken away with about
  two hours of notice. This repository and `notes/` are stored off the
  machine and survive. Everything else on the machine is lost.

## The rules of the measurement
- The engine is this repository with an executable `serve`. `serve <port>`
  must answer `GET /v1/models` when ready and `POST /v1/completions` with a
  prompt given as token ids. It returns its tokens as ids, either in
  `logprobs` or as `choices[0].token_ids`; the judge does not need the
  engine's own log-probabilities. If an executable `build` exists, it runs
  first.
- Smaller weights (4-bit and the like) are allowed. The reference engine
  runs the model's 16-bit weights, and AMD's tuned engine (below) is held to
  the same quality bar as yours.
- Quality is judged against a fixed bar, with two measures. A: the
  reference engine reads along each of your answers; the share of your
  tokens that are among its top choices, and how much less likely it found
  your tokens than its own first choice. B: your engine and the reference
  answer the same fresh tasks with checkable answers (arithmetic, multiple
  choice), graded by rule; your accuracy is compared with the reference's.
  The judge measures speed only after the answers pass.
- The limits: top-5 share at least 0.98; mean gap at most 0.08 nats per token;
  for B, the one-sided 95% lower bound of (your accuracy minus the
  reference's) at least -0.15.
- A submission passes when every judged request is answered and all three
  limits hold. The bar replaces the earlier rule that the answers match the
  reference's within a fixed tolerance; that agreement is still shown, not
  applied.
- The score is for one request at a time: the sum of the output tokens over
  the sum of the whole-request seconds, prompt time included. The timed
  requests are the same requests whose answers are checked. `./check`,
  `./submit` and `./score` also show the decode speed and the prompt time
  apart; those come from extra requests and do not count.
- The judge draws fresh prompts from a hidden pool for each submission. A
  drawn prompt is never sent to an engine in a later submission.
- The judge keeps the best engine that passed. A worse later submission
  does not replace it.
- The judge runs under another account. During `./submit`, `build` and
  `serve` run as the account `cand`, which cannot read the judge's files.
  `cand` reaches only this machine's loopback, not the network.
- Docker: if the repository has a `Dockerfile`, `./check` and `./submit`
  build it first, with no network during the build, and `serve` finds the
  image's name in `$IJ_ENGINE_IMAGE`. Containers start only through
  `sudo /opt/ij-amd/amd/controller/box/ij-docker run` (`--help` lists the
  options it takes): your own account's user id, the host's network, the
  GPU devices, the model read-only at `/models`, no other host directory
  and no added privileges. `/opt/ij-amd/amd/examples/amd-tuned-engine` on
  the GPU machine is an example.
- In its container a judged engine sees `/srv/models` read-only at `/models`
  and no other folder of the machine, and no network beyond this machine's
  loopback. Smaller weights are therefore made at start-up from those, or
  come with the commit or with the image named in `FROM`. `serve` must
  answer `GET /v1/models` within 30 minutes of its start.

## Commands
| Command | What it does | Time | Uses the GPU |
|---|---|---|---|
| `./gpu` | Shows who holds the GPU and who waits. | about 2 s | no |
| `./ref <prompt>` | Returns the reference engine's answer and token probabilities. | about 2 s if cached; 88 s measured when the reference had to start | on a cache miss |
| `./try <cmd>` | Runs your command on the GPU when it is free. | yours | yes |
| `./profile <cmd>` | Like `./try`, under rocprofv3. Prints the command's wall time, the kernel span, the kernels with the most time and amd-smi samples; the command's output goes to a log file. Hardware counters only with `--pmc`; whether this virtual GPU gives them is not confirmed. Processes inside Docker are not traced. | yours, plus a few seconds | yes |
| `./check` | Checks answers on a small public prompt set. Prints each failure and the speed in parts. | 3 min 12 s measured, before the speed parts were added (they add two more passes over the prompts) | yes |
| `./box <cmd>` | Runs your command on the GPU machine without taking the GPU (compiling, looking at the machine). | yours | no |
| `./submit` | Sends the current commit to the judge. Uses prompts drawn fresh from a hidden pool. | 9 min 03 s measured on 2026-11-18 for plain ATOM; longer when the engine is slower or the machine's reference speed is measured first | yes, alone |
| `./score` | Shows every submission: passed or failed, speed, and the two reference speeds below. `./score N` lists the requests of submission N. | under 1 s | no |

- There is one GPU. Commands that use it wait in one queue.
- `./submit` takes the GPU alone. All other GPU work waits until it ends.
- `./try`, `./profile`, `./check` and `./box` first copy your working tree
  to the GPU machine (not `.git`, not files that `.gitignore` excludes;
  such files made on the GPU machine stay there). Each working tree has its
  own copy there.

## Reference speeds (shown by `./score`)
- The reference engine on this machine: AMD's ATOM from its official image,
  pinned by digest, with one patch that only adds a logprob entry for every
  generated token (marker tokens such as `<think>` included).
- AMD's tuned engine for this model, measured on this machine. It is public:
  repository `proaditya/ATOM`, branch `qwen38-megakernels`. It is the
  fastest known engine for this model on this GPU.
- `./score` shows both speeds as the judge measured them on each GPU
  machine, by the same formula as the score. On the machine of 2026-11-18
  they were about 65 tokens/s for the reference engine and about 260 for
  the tuned engine.
- Published speeds of 700 to 1,000 tokens/s for the tuned engine are decode
  speeds. They leave out the prompt time, so they cannot be compared with
  the score, which includes it.
- The memory-bound speed of named configurations on this GPU: a measured
  copy rate divided by the size of the weights, for BF16 weights and for
  4-bit weights, each read once per token, one request, no speculative
  decoding. The copy rate is 4,700 GB/s, read plus write, from a plain copy
  kernel on this virtual GPU; it is a measurement, not the GPU's limit.
  These speeds are facts about those configurations, not limits on every
  engine: an engine that reads fewer bytes per token (smaller weights,
  speculative decoding) can be faster.

## Models
Two models are available: `anthropic/claude-opus-5-5` and `anthropic/claude-sonnet-5-5`.
`rlm.spawn` takes `model` and `thinking`; choose them per task. The main
agent runs on the first at `high`; a child gets its parent's level unless
`thinking` sets one. The reasoning levels, from least to most:
- `off`: no reasoning; the fastest and cheapest.
- `minimal`: almost none; nearly as fast and cheap as `off`.
- `low`: a little; fast and cheap.
- `medium`: moderate effort, speed and cost.
- `high`: more effort; slower and costlier.
- `xhigh`: more than `high`; slower and costlier again.
- `max`: the most effort; the slowest and costliest.
Prime Agent's model catalog gives these two models no setting of their own
for `off` and `minimal`.

## The end of the run
There is no finish line. The run ends when the operator stops it.

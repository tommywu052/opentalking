# FlashTalk 14B Realtime Optimization (Linux/WSL2 + Single Blackwell GPU)

> Status: 落地 · 2026-05  
> Hardware: NVIDIA RTX PRO 6000 Blackwell (96 GB, SM 12.0), single GPU  
> Stack: Linux/WSL2 (Ubuntu 22.04) · PyTorch 2.12 nightly · CUDA 12.8 · torchao FP8a8 + monkey-patch · `torch.compile(mode="reduce-overhead")`

This document captures the optimization journey from the initial Windows baseline to the
current near-realtime single-GPU pipeline, the dead-ends we explored, and the design
decisions that survived. Read it before tweaking any of the FlashTalk runner internals,
prebuffer code, or env vars.

---

## TL;DR — final numbers

| Stage | Per-chunk gen | Realtime ratio | First-sound latency | Notes |
|---|---|---|---|---|
| Windows BF16 + compile (baseline) | ~2.13 s | 1.90x | ~10 s | torchao broken on Win |
| Linux BF16 + compile | 1.35 s denoise | — | — | Validated WSL2 path |
| **Linux FP8a8 + compile + monkey-patch** | **0.97 s denoise** / **1.75 s/chunk** | **1.56x** | — | Bench (768×432) |
| + LLM 50-char cap, opener fast-path | varies | — | breaks A/V sync | Reverted |
| + adaptive prebuffer + 768×432 + opener-on | 1.83 s/chunk | 1.63x | ~6.3 s | Smooth |
| + opener off + 640×384 (current) | **1.31 s/chunk** | **1.17x** | **~4.8 s** | Smooth, clean replies |
| + dynamic T_gen ratio (current code) | 1.31 s/chunk | 1.17x | **~3.5 s** (predicted) | Auto-tunes any res / multi-GPU |

> "Chunk" = 28 video frames + 17 920 audio samples = **1.12 s of playback**.  
> `1.31 s/chunk` therefore means **1.17 × realtime** (still needs prebuffer).  
> `< 1.12 s/chunk` would be true realtime (no buffer needed) — achievable with a 2nd GPU
> via tensor parallelism.

---

## Why we left Windows for WSL2

Two hard blockers on Windows:

1. **torchao FP8a8 dispatch crash** in `_dispatch__torch_function__` when the W8A8
   tensor subclass interacted with `torch.compile`. Reproduced on torch 2.10/2.11/2.12
   nightly all the same way — environment-agnostic, it was a Windows
   ABI / platform-specific path in torchao itself.
2. **`torch.compile` "graph break / recompile_limit hit"** spam from VAE causal-3D
   convolutions and `einops.rearrange` shape unwrapping (`aten._local_scalar_dense`).
   Inductor on Windows kept evicting kernels mid-stream, causing 3-4 s spikes inside
   live generate.

Both vanish on Linux. WSL2 gives us native Linux PyTorch (no Windows path) while
keeping the model files on `/mnt/h` (Windows H: drive) and source on `/mnt/c`. The
production launcher copies `scripts/_wsl_run_backend.sh` → `~/opentalking-wsl/run_backend.sh`
(strip CRLF), then `bash run_backend.sh` exec's `apps.unified.main` on `0.0.0.0:18000`.

---

## The torchao monkey-patch (the unlock)

Even on Linux, `torchao`'s FP8 dynamic-activation path triggers a
`__torch_function__` recursion when tensor subclass meets compiled
graph. We patched it once at import time:

```python
# src/opentalking/models/flashtalk/torchao_patch.py
# Replace torchao's _dispatch__torch_function__ with a non-recursive
# variant that delegates back to the native ATen op when no override
# is registered, instead of calling into the subclass's own
# __torch_function__ again. Ships with FlashTalkPipeline init.
```

With the patch + `OPENTALKING_FLASHTALK_QUANTIZE=fp8a8` +
`OPENTALKING_FLASHTALK_COMPILE_QUANTIZED=1`, single-GPU benches:

```
Windows BF16+compile : 2.13 s/denoise
Linux   BF16+compile : 1.35 s/denoise  (-37%)
Linux   FP8a8+compile+patch : 0.97 s/denoise  (-54%)
```

Decode (VAE) accounts for the rest of the per-chunk time
(`0.63 s` at 768×432, `0.49 s` at 640×384). Total chunk is
denoise + decode + colour-correct + motion-encode ≈ 1.75–1.83 s @
768×432 or 1.27–1.31 s @ 640×384.

---

## Inductor recompile cap

VAE decoder iterates ~20 distinct CausalConv3d / ResidualBlock instances,
each with slightly different shape signatures. Default
`torch._dynamo.config.recompile_limit = 8` lets most of them fall back
to eager (we observed `[46/8] hit config.recompile_limit (8)`
→ ~0.66 s VAE decode). Bumping to 64 keeps everything compiled:

```bash
# scripts/_wsl_run_backend.sh (line 45)
export OPENTALKING_DYNAMO_RECOMPILE_LIMIT=64
```

---

## Bugs fixed during integration

| Symptom | Root cause | Fix |
|---|---|---|
| LLM repeated the same phrase ("小智！想听星星掉进咖啡杯的奇遇故事吗？") | `_llm_feeder` made a redundant *second* LLM call against `self.conversation`, so the first response leaked back into history and the next turn saw it as a "previous reply". | Bypass the second call when caller already provides `text`. The runner now replays the caller-supplied text into the splitter directly. |
| VAE graph breaks (`aten._local_scalar_dense.default`) → recompile spam | `einops.rearrange` triggers a Python int extraction inside compiled graph. | Rewrote rearrange with native `tensor.reshape().permute()` (preserves shape symbolically). |
| Mid-stream 3-4 s "denoise" spikes | Periodic `torch.cuda.empty_cache()` + `gc.collect()` were evicting Inductor's kernel cache. | Removed `_torch_gc()` calls inside the live loop. Memory headroom is plenty (96 GB Blackwell). |
| Sustained slowdown during long sentences | FlashTalk realtime ratio (T_gen > T_play) catches up on responses longer than the prebuffer. | LLM cap to 50 chars + adaptive prebuffer. |

---

## LLM length cap

50-char soft cap with explicit-detail exception (200 chars when user
asks for "详细说明 / 详细讲讲 / 详细介绍 / 多说一点 / 展开讲"):

```env
# .env
OPENTALKING_LLM_SYSTEM_PROMPT=你是一个友好的数字人小助手。【回答长度规则】默认每次回答总字数不超过50个字（包含标点和符号），简短直接命中问题核心，不展开、不举例、不分点列举、不使用项目符号。【例外】当用户明确说"详细说明 / 详细讲讲 / 详细介绍 / 多说一点 / 展开讲"等类似要求时，可以放宽到200字以内并适度展开。语气自然亲切，像朋友一样轻松地说话，不要用Markdown格式。
```

History: tried 30 (too curt for natural chat), then 50 with exception
escape hatch. The 200-char "detail" path triggers a 7-15 chunk
response, which is exactly what the static prebuffer ceiling
(`FLASHTALK_PREBUFFER_CHUNKS=4`) is sized for.

---

## TTS opener — three variants, all wrong, all worth documenting

The original goal: hide ASR→first-sound latency by playing a cached
short interjection ("嗯嗯", "好的", "OK", "明白") immediately while
the real LLM/TTS/FlashTalk pipeline runs in parallel.

### Variant 1 — pure fast-path

Start pacing on the first opener chunk. Real chunks stream in as
they arrive after the opener.

**Failure**: the realtime deficit (`T_gen - T_play = 0.71 s` @ 768×432)
applies to every subsequent chunk. User got smooth opener, then
sustained stutter (0.7 s gap between every real chunk).

### Variant 2 — fast-path + small "real prebuffer" (Option B hybrid)

Opener still starts pacing immediately, but hold the first 2 real
chunks in `real_pending` and flush them together once both ready.

**Failure**: created a mid-stream **2.3 s silent gap** between
opener-end and real-content-start. WebRTC RTP timestamps of the
post-gap frames pointed to wall times **already 2 s in the past**;
receiver's jitter buffer (~200 ms tolerance) dropped/delayed frames
non-uniformly across audio vs video tracks, producing **lip-sync
drift**. User report: "嘴型和影音是分離的" — confirmed via timing
breakdown in the speak pipeline log.

### Variant 3 — opener as part of prebuffer

Opener counts toward `FLASHTALK_PREBUFFER_CHUNKS` like any other
chunk. RTP stream stays continuous; A/V solid.

**Failure mode**: opener stops contributing to *latency reduction* —
its only remaining function is variety ("嗯嗯/好的" prefix). User
correctly observed this added words to every reply with no benefit.

### Conclusion: opener disabled

```bash
# scripts/_wsl_run_backend.sh (line 78)
export OPENTALKING_FLASHTALK_TTS_OPENER_ENABLE=0
```

The machinery (rule-based opener selection, padded PCM cache, per-session
TTS-provider plumbing for Qwen voice clones) stays in
`flashtalk_runner.py` for future revisits but is gated by the env var.

> **Pitfall**: the WSL-side launcher `~/opentalking-wsl/run_backend.sh`
> is a **copy** of the workspace `scripts/_wsl_run_backend.sh`. The
> first attempt to disable opener edited only the workspace copy, so
> the live backend kept reading `ENABLE=1`. The launcher is now
> synced + `dos2unix`'d as part of every restart.

---

## Adaptive prebuffer (the survivor)

### What prebuffer does and why

Static `FLASHTALK_PREBUFFER_CHUNKS=N` holds N chunks before starting
WebRTC pacing. This is unavoidable when `T_gen > T_play`: each
generated chunk arrives `T_gen - T_play` seconds late relative to the
playback consumption rate, so the buffer absorbs the cumulative
deficit.

Derivation:

```
At pacing start: queued = N chunks, played = 0
After time t since pacing:
    played   = t / T_play
    arrived  = N + t / T_gen
Need always: arrived ≥ played
    N + t/T_gen ≥ t/T_play
    N ≥ t × (T_gen - T_play) / (T_play × T_gen)
Worst case t = M × T_play (M = total chunks):
    N ≥ M × (T_gen - T_play) / T_gen
    N ≥ M × (1 - T_play/T_gen)
```

`(1 - T_play/T_gen)` = "deficit ratio". For 768×432 it's 0.39; for
640×384 it's 0.15.

### Static N wastes latency on short replies

50-char reply ≈ 4-5 chunks. Static N=4 makes pacing wait 4 × 1.83 ≈
7.3 s of FlashTalk even though formula says 2 chunks would suffice
(`ceil(0.39 × 5) + 1 safety = 3`). For 50-char + 640×384 (`0.15 × 5`)
the formula gives 2.

### Implementation: producer signals total, consumer measures T_gen

Producer (the LLM→TTS→audio_q stage) increments
`producer_chunks_pushed[0]` per push and sets `producer_done_evt`
right before the `None` sentinel. Consumer reads both:

```2125:2155:src/opentalking/worker/flashtalk_runner.py
                # ADAPTIVE PREBUFFER (auto-tuned by measured T_gen)
                # ------------------------------------------------
                # FlashTalk has a "realtime deficit": each chunk takes
                # T_gen seconds to generate but only T_play (= 1.12 s)
                # of video plays back. ...
                # We measure flashtalk_gen_sum_s / generated after each
                # chunk and recompute on the fly.
                T_PLAY_SEC = chunk_samples / sample_rate  # 1.12 s
```

```2207:2226:src/opentalking/worker/flashtalk_runner.py
                    # Adaptive prebuffer target (uses measured T_gen).
                    if producer_done_evt.is_set():
                        expected_total = producer_chunks_pushed[0]
                        if expected_total > 0 and generated > 0:
                            avg_t_gen = flashtalk_gen_sum_s / generated
                            deficit_ratio = max(
                                0.0,
                                (avg_t_gen - T_PLAY_SEC) / avg_t_gen,
                            )
                            required = max(1, math.ceil(deficit_ratio * expected_total) + 1)
                            target_prebuffer = min(required, prebuffer_chunks)
                        else:
                            target_prebuffer = 1
                    else:
                        target_prebuffer = prebuffer_chunks
```

Behaviour:

- Producer typically finishes within ~2 s (TTS is the long pole).
- FlashTalk needs ≥ 2 × T_gen for the smallest meaningful prebuffer.
- → `producer_done_evt` is reliably set by the time consumer evaluates
  prebuffer, so the formula path activates from chunk 2 onward.
- If producer hasn't finished yet (very long response), fall back to
  static `prebuffer_chunks` ceiling — the worst-case-safe behaviour.
- Per-iteration `have_all` shortcut: if `generated >= producer_chunks_pushed[0]`
  (response shorter than computed target), flush immediately.

Live log line for diagnosis:

```
Pre-buffer done (3/3 chunks; 0 opener + 3 real, expected_total=4,
                avg_T_gen=1.31s, T_play=1.12s), starting pacing
```

### Why dynamic T_gen (and not a hard-coded ratio)

We hardcoded `0.388` initially (correct for 768×432). Then resolution
dropped to 640×384 making it 0.15 — same code suddenly over-buffered.
Measuring T_gen live makes the formula self-tune across:

- Resolution changes (640×384 ↔ 768×432)
- Quantization changes (BF16 ↔ FP8a8 ↔ INT8)
- **Multi-GPU**: with 2× model-parallel, T_gen drops below T_play
  (deficit ratio = 0) → `target_prebuffer = 1` → effectively no buffer.

---

## Resolution downscale: 768×432 → 640×384

Pixel count drops 26%, denoise time drops ~28% (linear in latents).

| Resolution | Pixels | T_gen | Realtime ratio | Notes |
|---|---|---|---|---|
| 768×432 | 331 776 | 1.83 s | 1.63x | Original |
| **640×384** | **245 760** | **1.31 s** | **1.17x** | **Current** |
| 512×288 | 147 456 | ~1.0 s (est) | ~0.9x | Visible quality drop |

> **Latent shape constraint**: width / height must be divisible by **16**
> (VAE 8× spatial compression × transformer patch_size 2×2). 360 fails
> (`360 / 16 = 22.5`); 384 works (`384 / 16 = 24`). Aspect 1.67 vs
> original 1.71 — barely noticeable on talking-head.

```49:50:scripts/_wsl_run_backend.sh
export OPENTALKING_FLASHTALK_HEIGHT=640
export OPENTALKING_FLASHTALK_WIDTH=384
```

First time the model runs at a new resolution, Inductor recompiles
VAE + WanModel kernels (~2-3 minutes). Subsequent sessions hit the
cache (`$TORCHINDUCTOR_CACHE_DIR=$HOME/opentalking-wsl/.inductor-cache`).

---

## Pipeline overlap

Three stages run concurrently via `asyncio.gather`:

```
Speak start
  ├─ Producer  : LLM stream → sentence split → TTS (per sentence, queued)
  │              → fixed-size PCM chunks → audio_q
  └─ Consumer  : audio_q → FlashTalk frame gen → prebuffer check → WebRTC
```

Observed timing (640×384, 17-char reply, 5 chunks):

```
speak_wall_ms       =  9_787   (parallel total)
├─ tts_worker_wall  =  1_478   (TTS finishes here, producer_done_evt set)
├─ producer_wall    =  2_100
└─ flashtalk_gen_sum= 11_540 / 7 chunks ≈ 1.65 s/chunk avg
   first_chunk      =   621 ms (first PCM in audio_q)
   first_flashtalk  = 2_523 ms (first frames generated)
   first_webrtc     = 4_815 ms (pacing starts; this is "first sound")
```

`first_webrtc_queue_ms` is the metric that maps to user-perceived
"ASR done → first sound" delay. Plus ~500 ms of frontend pacing on
top.

---

## Remaining levers (not pulled)

| Lever | Estimated gain | Effort | Quality risk |
|---|---|---|---|
| **+1 RTX PRO 6000 GPU** (model parallel) | T_gen → ~0.6-0.7 s, ratio < 1, prebuffer = 1, first sound ~1 s | hardware | none |
| INT4 weight-only (GPTQ/AWQ) | T_gen 1.3 → ~1.0 s | calibration data + retesting | mild on motion detail |
| xFuser fused attention kernels | T_gen -10-15% (single-GPU) | repackage inference stack | none |
| Resolution → 512×288 | T_gen → ~1.0 s | env change | visible blurring |
| Speculative TTS (start TTS on partial ASR) | ~0.3-0.5 s if gambles right | medium | re-render on bad guess |

The clean path forward is **second GPU**. With model-parallel xFuser
the deficit ratio goes negative (T_gen < T_play) and the entire
prebuffer mechanism becomes a no-op — first sound drops to whatever
TTS+1 chunk takes (~1 s).

---

## Operational notes

### Restarting the backend

```powershell
# from PowerShell
wsl -d Ubuntu-22.04 -- bash -lc "cd ~/opentalking-wsl && bash ./run_backend.sh"
```

Launcher self-kills any stale process on `:18000` before binding.
Cold start: ~5 min model load (slow `/mnt/h` I/O), then VAE + WanModel
compile ~2-3 min on first session. Warm: ~10 s startup.

### Syncing launcher edits to WSL

The active launcher is `~/opentalking-wsl/run_backend.sh`, copied
from `scripts/_wsl_run_backend.sh`. Always sync after editing:

```bash
wsl -d Ubuntu-22.04 -- bash -lc \
  "cp /mnt/c/Nvidia/opentalking/scripts/_wsl_run_backend.sh \
       ~/opentalking-wsl/run_backend.sh && \
   sed -i 's/\r\$//' ~/opentalking-wsl/run_backend.sh"
```

The `sed` step strips Windows CRLF — without it bash chokes on
`set -euo pipefail`.

### Diagnosing latency

Look for the per-speak summary line:

```
Speak pipeline timing: session=... speak_wall_ms=... opener_ms=...
  llm_tts_gather_ms=... producer_wall_ms=... flashtalk_generate_sum_ms=...
  flashtalk_chunks=... first_webrtc_queue_ms=... response_chars=...
```

`first_webrtc_queue_ms` is the single most important metric. Together
with the new `Pre-buffer done (... avg_T_gen=Xs ...)` line you can
infer whether GPU throughput regressed (T_gen up) or response just
got long (chunks up).

---

## Configuration reference (current production)

```bash
# .env / launcher highlights
OPENTALKING_FLASHTALK_QUANTIZE=fp8a8
OPENTALKING_FLASHTALK_COMPILE_QUANTIZED=1
OPENTALKING_FLASHTALK_HEIGHT=640
OPENTALKING_FLASHTALK_WIDTH=384
OPENTALKING_FLASHTALK_SAMPLE_STEPS=1
OPENTALKING_DYNAMO_RECOMPILE_LIMIT=64
OPENTALKING_FLASHTALK_TTS_OPENER_ENABLE=0
FLASHTALK_PREBUFFER_CHUNKS=4               # static ceiling, adaptive logic shrinks dynamically
```

End of document.

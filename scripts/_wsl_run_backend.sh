#!/usr/bin/env bash
# Launch OpenTalking unified backend inside WSL2 with the
# Linux + torch 2.12 + FP8a8 + compile + torchao patch stack.
#
# Backend listens on 0.0.0.0:18000 — Windows browser can hit
# http://localhost:18000 directly (WSL2 NAT auto-forwards localhost).
#
# Usage (from PowerShell):
#   wsl -d Ubuntu-22.04 -- bash -lc 'bash ~/opentalking-wsl/run_backend.sh'
set -euo pipefail

# 1) activate venv (native Linux fs)
source ~/opentalking-wsl/.venv/bin/activate

# 2) source code on /mnt/c, Inductor cache on native fs (re-use bench cache)
export PYTHONPATH=/mnt/c/Nvidia/opentalking/src
export TORCHINDUCTOR_CACHE_DIR="$HOME/opentalking-wsl/.inductor-cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# 3) translate Windows model paths → /mnt/h
export OPENTALKING_FLASHTALK_MODE=local
export OPENTALKING_FLASHTALK_CKPT_DIR=/mnt/h/opentalking-models/SoulX-FlashTalk-14B
export OPENTALKING_FLASHTALK_WAV2VEC_DIR=/mnt/h/opentalking-models/chinese-wav2vec2-base
# HF cache (Wav2Vec2 etc.)
export HF_HOME=/mnt/h/opentalking-cache/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/h/opentalking-cache/huggingface
# Idle cache
export OPENTALKING_FLASHTALK_IDLE_CACHE_DIR=/mnt/h/opentalking-cache/idle

# 4) FP8a8 + compile + torchao dispatch patch
#    These are the settings that gave 0.97s/denoise & 1.75s/chunk in bench.
export OPENTALKING_FLASHTALK_QUANTIZE=fp8a8
export OPENTALKING_FLASHTALK_COMPILE_QUANTIZED=1
export OPENTALKING_FLASHTALK_DISABLE_COMPILE=0
# Render resolution. FlashTalk realtime ratio scales roughly linearly
# with pixel count (denoise dominates):
#
#   768×432 = 331 776 px  → 1.83 s/chunk → 1.63x realtime (needs prebuffer)
#   640×384 = 245 760 px  → ~1.40 s/chunk (est) → ~1.25x realtime
#   512×288 = 147 456 px  → ~1.0 s/chunk (est) → ~0.9x realtime (no buffer)
#
# IMPORTANT: width/height MUST be divisible by 16 (VAE compression 8×
# + transformer patch 2×2). 640×360 fails because 360 / 16 = 22.5.
# 640×384 (aspect 1.67 vs original 1.71) is the nearest legal
# downscale; visual quality drop is mild on talking-head crops.
#
# Lower latency target: 640×384 should drop first-sound from ~6 s to
# ~4 s with adaptive prebuffer, while keeping smooth playback.
export OPENTALKING_FLASHTALK_HEIGHT=640
export OPENTALKING_FLASHTALK_WIDTH=384
export OPENTALKING_FLASHTALK_SAMPLE_STEPS=1
export OPENTALKING_FLASHTALK_SAMPLE_RATE=16000
export OPENTALKING_FLASHTALK_INIT_TIMEOUT_SEC=900   # cold start can take 5-10 min on /mnt/h

# Bump torch._dynamo recompile cap. The VAE decoder iterates through ~20
# different CausalConv3d / ResidualBlock instances; default 8 lets most of
# them fall back to eager (we saw `[46/8] hit config.recompile_limit (8)`
# in the backend log → ~0.66 s VAE decode). 64 covers all real shapes.
export OPENTALKING_DYNAMO_RECOMPILE_LIMIT=64

# Static (worst-case) pre-buffer ceiling. Adaptive logic in the
# consumer reduces this dynamically once TTS finishes producing audio:
#
#   required_prebuffer = ceil(0.388 × total_chunks) + 1
#   actual_prebuffer   = min(required_prebuffer, FLASHTALK_PREBUFFER_CHUNKS)
#
# The 0.388 factor is the FlashTalk realtime deficit (1.83 s gen / 1.12 s
# play → 0.71 s deficit / 1.83 s gen ≈ 0.388). For typical 50-char
# responses (~5 chunks) the formula yields prebuffer=3, dropping
# first-sound latency from 4×1.83=7.3 s to 3×1.83=5.5 s. For long
# responses (≥10 chunks, e.g. user explicitly requesting detail) the
# formula returns 4 = static cap, preserving the old smooth-but-slow
# behaviour. See consumer comments in flashtalk_runner.py.
export FLASHTALK_PREBUFFER_CHUNKS=4

# ---- TTS opener — DISABLED ----
#
# History: opener was originally added as a "latency-hiding fast-path"
# (start pacing on the first cached opener chunk so user hears "嗯嗯"
# while the real response was generating). That broke WebRTC A/V
# sync (mid-stream RTP gap). Routing opener through the prebuffer
# fixed sync but eliminated the latency benefit, leaving opener as a
# pure variety add-on that prepended 1-3 chars to every reply.
#
# With adaptive prebuffer (above) we get the latency reduction
# without the opener mechanism, so we just turn it off — replies
# stay short, no extra "嗯嗯/好的/OK/明白" prefix.
#
# (The opener machinery is still in flashtalk_runner.py in case we
# want to revisit; setting ENABLE=0 is enough to bypass it at
# runtime.)
export OPENTALKING_FLASHTALK_TTS_OPENER_ENABLE=0

# 5) bind to 0.0.0.0 so Windows browser can reach this backend
export OPENTALKING_API_HOST=0.0.0.0
export OPENTALKING_API_PORT=18000
export OPENTALKING_UNIFIED_HOST=0.0.0.0
export OPENTALKING_UNIFIED_PORT=18000
export OPENTALKING_UNIFIED_UVICORN_WORKERS=1

# 6) misc
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# 7) sanity: stale backend on :18000?
PORT_PID=$(lsof -ti :18000 2>/dev/null || true)
if [[ -n "$PORT_PID" ]]; then
    echo "[port] killing stale PID $PORT_PID on :18000"
    kill -9 $PORT_PID || true
    sleep 2
fi

cd /mnt/c/Nvidia/opentalking
echo "============================================================"
echo " OpenTalking unified backend (WSL2 / Linux / FP8a8+compile)"
echo "============================================================"
echo "  python : $(which python)"
echo "  pwd    : $(pwd)"
echo "  ckpt   : $OPENTALKING_FLASHTALK_CKPT_DIR"
echo "  cache  : $TORCHINDUCTOR_CACHE_DIR"
echo "  bind   : 0.0.0.0:18000  (browser → http://localhost:18000)"
echo "  quant  : $OPENTALKING_FLASHTALK_QUANTIZE"
echo "  compile: ON (compile_quantized=$OPENTALKING_FLASHTALK_COMPILE_QUANTIZED)"
echo "------------------------------------------------------------"
echo " Cold start: model load ~5min (slow /mnt/h I/O), then VAE +"
echo " WanModel compile ~2-3min on first session. After warmup,"
echo " each chunk is ~1.75s (vs ~2.13s on Windows)."
echo "============================================================"

exec python -m apps.unified.main

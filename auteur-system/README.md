# The Auteur System

High-Performance Video Generation AI Pipeline for NVIDIA DGX Spark (GB10).

## Verified Architecture (2025-02-01)

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE AUTEUR SYSTEM                            │
│              Video Generation for Grace Blackwell               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    MODEL TIERS (Verified)               │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  validator    │ SVD-XT (I2V)    │ FP16  │ VERIFIED     │   │
│  │  workhorse    │ HunyuanVideo    │ BF16  │ VERIFIED     │   │
│  │  experimental │ CogVideoX       │ BF16  │ UNSTABLE     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 OPTIMIZATION MODES                      │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Safe Mode     │ CPU Offload ON  │ Shared workloads    │   │
│  │  Speed Mode    │ --no-offload    │ 33% faster (bench)  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Benchmark Results

**Platform:** NVIDIA GB10 (Grace-Blackwell), 128GB LPDDR5X Unified Memory

| Mode | Time (17 frames, 20 steps) | Peak RSS | CUDA Allocated | Speedup |
|------|---------------------------|----------|----------------|---------|
| Baseline (offload ON) | 568.70s | 41.74 GB | ~0 GB (offloaded) | - |
| **Speed Run (offload OFF)** | **380.06s** | 29.11 GB | 41.40 GB | **33% faster** |

### Key Findings

1. **CPU offloading hurts latency on GB10** - On unified memory, offload hooks add software overhead without physical benefits.
2. **Model fits comfortably** - HunyuanVideo uses ~37GB, leaving ~90GB headroom.
3. **Attention backend: SDPA** - PyTorch native SDPA is the effective backend (FlashAttention is installed but Diffusers uses SDPA).

## Quick Start

```bash
# Build the container
cd ~/auteur-system
docker compose build

# Run with CPU offloading (safe mode - for shared workloads)
docker compose run auteur-pipeline python generate.py \
  --model-tier workhorse \
  --prompt "A sunset over mountains"

# Run WITHOUT offloading (speed mode - 33% faster)
docker compose run auteur-pipeline python generate.py \
  --model-tier workhorse \
  --prompt "A sunset over mountains" \
  --no-offload

# Image-to-Video (SVD)
docker compose run auteur-pipeline python generate.py \
  --model-tier validator \
  --input-image /app/assets/svd_input_1024x576.jpg
```

## Command Line Options

```
--model-tier {validator,workhorse,experimental}
    validator:    SVD-XT (Image-to-Video) - Fast, reliable
    workhorse:    HunyuanVideo (Text-to-Video) - Production quality
    experimental: CogVideoX (Text-to-Video) - UNSTABLE, dtype issues

--no-offload      Disable CPU offloading (GB10 speed mode)
--compile         Enable torch.compile with max-autotune
--monitor-bandwidth  Track memory usage during generation
--smoke-test      Run rigorous hardware verification
```

## Memory Profile (HunyuanVideo)

| Component | Size | % of Total |
|-----------|------|------------|
| transformer (DiT) | 23.88 GB | 65.1% |
| text_encoder (CLIP) | 12.04 GB | 32.8% |
| text_encoder_2 (LLaMA) | 0.28 GB | 0.8% |
| vae | 0.46 GB | 1.3% |
| **TOTAL** | **36.67 GB** | - |

## Hardware Requirements

- NVIDIA DGX Spark (GB10) or compatible Grace Blackwell system
- ARM64 architecture (aarch64)
- 128GB Unified Memory
- ConnectX-7 for multi-node (optional)

## Software Stack

- **Base Image:** `nvcr.io/nvidia/pytorch:25.10-py3`
- **CUDA:** 13.0.2 (matches DGX Spark toolkit)
- **Compute Capability:** sm_121 (Blackwell)
- **Included:** Transformer Engine, DALI, nvImageCodec, Torch-TensorRT
- **Attention:** PyTorch SDPA (FlashAttention 2.7.4.post1 installed)

## Known Issues

1. **Video file size:** Some outputs are unexpectedly small (~30KB). This appears to be an encoding issue with `export_to_video`, not a generation issue.
2. **CogVideoX dtype conflicts:** T5 encoder requires FP32 while Apex fused RMSNorm causes failures.

## Metrics Tracked

The system tracks:
- **Peak RSS (VmHWM)** - Accurate for unified memory systems
- **CUDA Allocated** - GPU memory allocation
- **ffprobe validation** - Output video verification (dimensions, frames, duration)

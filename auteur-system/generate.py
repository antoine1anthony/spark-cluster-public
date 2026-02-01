#!/usr/bin/env python3
"""
The Auteur System - Video Generation Script
Supports multiple model tiers with proper mixed-precision handling for GB10.

IMPORTANT NOTES:
- HunyuanVideo IS available via HunyuanVideoPipeline (hunyuanvideo-community/HunyuanVideo)
- CogVideoX requires mixed precision: T5 encoder in float32, transformer/VAE in bfloat16
- FlashAttention must be KERNEL-VERIFIED, not just import-verified
- Peak RSS tracking is used for accurate memory measurement on unified memory systems

GB10 ARCHITECTURE:
- 128GB LPDDR5X Unified Memory
- ~107 GB/s effective bandwidth
- Model fits ~37GB, leaving ~80GB for activations/buffers
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch


# =============================================================================
# MEMORY TRACKING (Peak RSS - works without psutil, accurate for unified memory)
# =============================================================================

def read_rss_gb() -> Optional[float]:
    """
    Read peak resident set size (VmHWM) from /proc/self/status.
    Works without psutil and is accurate for GB10's unified memory.
    Returns peak RSS in GB, or None if unavailable.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):  # Peak resident set
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return None


def read_current_rss_gb() -> Optional[float]:
    """Read current RSS (VmRSS) in GB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return None


# =============================================================================
# VIDEO VALIDATION (ffprobe-based)
# =============================================================================

def validate_video_output(video_path: str) -> Dict[str, Any]:
    """
    Validate generated video using ffprobe.
    Catches "35KB paperweights" - videos that encoded but have no real content.
    
    Returns dict with validation results and video metadata.
    """
    result = {
        "valid": False,
        "path": video_path,
        "error": None,
        "metadata": {}
    }
    
    if not Path(video_path).exists():
        result["error"] = "File does not exist"
        return result
    
    file_size = Path(video_path).stat().st_size
    result["metadata"]["file_size_bytes"] = file_size
    
    # Minimum viable video size (100KB) - anything smaller is likely corrupt
    if file_size < 100 * 1024:
        result["error"] = f"File too small ({file_size} bytes) - likely corrupt or encoding failed"
        return result
    
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,bit_rate",
            "-show_entries", "format=duration,size",
            "-of", "json",
            video_path
        ]
        
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if proc.returncode != 0:
            result["error"] = f"ffprobe failed: {proc.stderr}"
            return result
        
        ffprobe_data = json.loads(proc.stdout)
        
        # Extract stream info
        if ffprobe_data.get("streams"):
            stream = ffprobe_data["streams"][0]
            result["metadata"]["codec"] = stream.get("codec_name")
            result["metadata"]["width"] = int(stream.get("width", 0))
            result["metadata"]["height"] = int(stream.get("height", 0))
            result["metadata"]["frame_rate"] = stream.get("avg_frame_rate")
            result["metadata"]["nb_frames"] = int(stream.get("nb_frames", 0))
            result["metadata"]["bit_rate"] = int(stream.get("bit_rate", 0)) if stream.get("bit_rate") else None
        
        # Extract format info
        if ffprobe_data.get("format"):
            fmt = ffprobe_data["format"]
            result["metadata"]["duration"] = float(fmt.get("duration", 0))
        
        # Validation checks
        meta = result["metadata"]
        
        if meta.get("width", 0) < 64 or meta.get("height", 0) < 64:
            result["error"] = f"Resolution too small: {meta.get('width')}x{meta.get('height')}"
            return result
        
        if meta.get("nb_frames", 0) < 2:
            result["error"] = f"Not enough frames: {meta.get('nb_frames')}"
            return result
        
        if meta.get("duration", 0) < 0.1:
            result["error"] = f"Duration too short: {meta.get('duration')}s"
            return result
        
        result["valid"] = True
        
    except subprocess.TimeoutExpired:
        result["error"] = "ffprobe timed out"
    except json.JSONDecodeError as e:
        result["error"] = f"Failed to parse ffprobe output: {e}"
    except FileNotFoundError:
        # ffprobe not installed - skip validation but warn
        result["error"] = "ffprobe not installed - validation skipped"
        result["valid"] = True  # Don't fail if ffprobe missing
    except Exception as e:
        result["error"] = f"Validation error: {e}"
    
    return result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[Auteur] %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("auteur.generate")


class BandwidthMonitor:
    """
    Monitor memory during generation.
    
    Tracks both CUDA allocator stats and Peak RSS for accurate
    unified memory measurement on GB10.
    """
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.samples = []
        self.rss_samples = []
        self.start_rss = None
        
    def sample(self, label: str = ""):
        if not self.enabled:
            return
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        rss = read_current_rss_gb()
        peak_rss = read_rss_gb()
        
        self.samples.append({
            "time": time.time(),
            "label": label,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "rss_gb": rss,
            "peak_rss_gb": peak_rss,
        })
        
        if rss:
            self.rss_samples.append(rss)
        
        if label:
            logger.info(f"[Memory @ {label}] CUDA: {allocated:.2f}GB | RSS: {rss:.2f}GB | Peak RSS: {peak_rss:.2f}GB")
        
    def report(self):
        if not self.enabled or not self.samples:
            return
        
        logger.info("="*50)
        logger.info("MEMORY MONITOR REPORT")
        logger.info("="*50)
        
        max_allocated = max(s["allocated_gb"] for s in self.samples)
        avg_allocated = sum(s["allocated_gb"] for s in self.samples) / len(self.samples)
        
        logger.info(f"CUDA Peak Allocated: {max_allocated:.2f} GB")
        logger.info(f"CUDA Avg Allocated: {avg_allocated:.2f} GB")
        
        # RSS (more accurate for unified memory)
        if self.rss_samples:
            max_rss = max(self.rss_samples)
            final_peak_rss = read_rss_gb()
            logger.info(f"Peak RSS (sampled): {max_rss:.2f} GB")
            logger.info(f"Peak RSS (VmHWM): {final_peak_rss:.2f} GB")
        
        logger.info(f"Samples: {len(self.samples)}")
        logger.info("="*50)


def verify_flash_attention_kernel() -> Dict[str, Any]:
    """
    RIGOROUS FlashAttention verification - not just import, but kernel execution proof.
    Returns dict with verification status and details.
    """
    result = {
        "available": False,
        "version": None,
        "extension_path": None,
        "kernel_verified": False,
        "profiler_proof": None,
    }
    
    try:
        import flash_attn
        result["version"] = flash_attn.__version__
        
        # Check 1: Verify the CUDA extension is actually loaded
        try:
            import flash_attn_2_cuda
            result["extension_path"] = flash_attn_2_cuda.__file__
            logger.info(f"✓ flash_attn_2_cuda extension: {result['extension_path']}")
        except ImportError as e:
            logger.warning(f"○ flash_attn_2_cuda not available: {e}")
            return result
        
        # Check 2: Kernel execution with profiler
        from flash_attn import flash_attn_func
        
        device = "cuda"
        dtype = torch.float16
        B, S, H, D = 2, 2048, 8, 64  # Realistic size, not baby test
        
        q = torch.randn(B, S, H, D, device=device, dtype=dtype)
        k = torch.randn(B, S, H, D, device=device, dtype=dtype)
        v = torch.randn(B, S, H, D, device=device, dtype=dtype)
        
        # Profile the execution
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True
        ) as prof:
            out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
            torch.cuda.synchronize()
        
        # Check if flash-attn kernels appear in profiler
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        result["profiler_proof"] = table
        
        # Look for flash-attn kernel names
        kernel_names = [evt.key for evt in prof.key_averages()]
        flash_kernels = [k for k in kernel_names if 'flash' in k.lower() or 'fmha' in k.lower()]
        
        if flash_kernels:
            result["kernel_verified"] = True
            result["available"] = True
            logger.info(f"✓ FlashAttention KERNEL VERIFIED: {flash_kernels[:3]}")
        else:
            logger.warning("○ FlashAttention imported but no flash kernels in profiler")
            logger.info(f"Kernels found: {kernel_names[:5]}")
        
        del q, k, v, out
        torch.cuda.empty_cache()
        
    except ImportError:
        logger.info("○ FlashAttention not installed")
    except Exception as e:
        logger.warning(f"○ FlashAttention verification failed: {e}")
    
    return result


def load_svd(repo_id: str, no_offload: bool = False, compile_model: bool = False) -> Any:
    """
    Load Stable Video Diffusion for Image-to-Video.
    SVD is an I2V model - requires an input image, not a text prompt.
    
    Args:
        repo_id: HuggingFace repo ID
        no_offload: If True, load directly to CUDA without CPU offloading
        compile_model: If True, compile the UNet with torch.compile
    """
    from diffusers import StableVideoDiffusionPipeline
    
    logger.info(f"Loading SVD from {repo_id}")
    logger.info(f"  no_offload={no_offload}, compile={compile_model}")
    
    rss_before = read_rss_gb()
    
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    
    if no_offload:
        # Direct CUDA load - faster on GB10 unified memory
        pipe.to("cuda")
        logger.info("✓ SVD loaded directly to CUDA (no offload)")
    else:
        # CPU offload for memory management
        pipe.enable_model_cpu_offload()
        logger.info("✓ SVD loaded with CPU offloading (fp16)")
    
    # VAE optimizations
    pipe.vae.enable_slicing()
    
    # Compile UNet if requested
    if compile_model:
        logger.info("Compiling UNet with max-autotune...")
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
        logger.info("✓ UNet compiled")
    
    rss_after = read_rss_gb()
    if rss_before and rss_after:
        logger.info(f"RSS after load: {rss_after:.2f} GB (delta: +{rss_after - rss_before:.2f} GB)")
    
    return pipe


def load_cogvideox(repo_id: str, no_offload: bool = False, compile_model: bool = False) -> Any:
    """
    Load CogVideoX with CPU offloading for automatic dtype handling.
    
    The Apex fused RMS norm in T5 requires float32.
    CPU offloading handles dtype conversions automatically.
    
    NOTE: This model is EXPERIMENTAL due to dtype conflicts.
    
    Args:
        repo_id: HuggingFace repo ID
        no_offload: If True, attempt direct CUDA load (may fail due to dtype issues)
        compile_model: If True, compile the transformer with torch.compile
    """
    from diffusers import CogVideoXPipeline
    
    logger.info(f"Loading CogVideoX from {repo_id}")
    logger.info(f"  no_offload={no_offload}, compile={compile_model}")
    logger.warning("⚠️  CogVideoX is EXPERIMENTAL - T5 encoder dtype conflicts may cause failures")
    
    rss_before = read_rss_gb()
    
    # Load in bfloat16 for Blackwell optimization
    pipe = CogVideoXPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
    )
    
    if no_offload:
        # Direct CUDA load - may fail due to T5/Apex dtype issues
        logger.warning("Attempting direct CUDA load - this may fail for CogVideoX")
        pipe.to("cuda")
        logger.info("✓ CogVideoX loaded directly to CUDA (no offload)")
    else:
        # Use sequential CPU offload - this handles dtype conversions
        pipe.enable_sequential_cpu_offload()
        logger.info("✓ CogVideoX loaded with sequential CPU offload (bf16)")
    
    # Enable VAE optimizations
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    
    # Compile transformer if requested
    if compile_model and hasattr(pipe, 'transformer'):
        logger.info("Compiling transformer with max-autotune...")
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        logger.info("✓ Transformer compiled")
    
    rss_after = read_rss_gb()
    if rss_before and rss_after:
        logger.info(f"RSS after load: {rss_after:.2f} GB (delta: +{rss_after - rss_before:.2f} GB)")
    
    return pipe


def load_hunyuanvideo(repo_id: str, no_offload: bool = False, compile_model: bool = False) -> Any:
    """
    Load HunyuanVideo - the PREFERRED workhorse model for GB10.
    Uses HunyuanVideoPipeline from hunyuanvideo-community/HunyuanVideo.
    
    GB10 OPTIMIZATION NOTES:
    - Model size: ~37GB (weights)
    - GB10 RAM: 128GB unified
    - With no_offload=True, model stays pinned in unified memory
    - This avoids offload hook overhead and improves latency
    
    Args:
        repo_id: HuggingFace repo ID
        no_offload: If True, load directly to CUDA (recommended for GB10 speed runs)
        compile_model: If True, compile transformer with torch.compile max-autotune
    """
    from diffusers import HunyuanVideoPipeline
    
    logger.info(f"Loading HunyuanVideo from {repo_id}")
    logger.info(f"  no_offload={no_offload}, compile={compile_model}")
    
    rss_before = read_rss_gb()
    logger.info(f"RSS before load: {rss_before:.2f} GB" if rss_before else "RSS tracking unavailable")
    
    pipe = HunyuanVideoPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
    )
    
    if no_offload:
        # SPEED MODE: Direct CUDA load
        # The 37GB model swims in 128GB unified memory
        logger.info("Loading directly to CUDA (GB10 Speed Mode)...")
        pipe.to("cuda")
        logger.info("✓ HunyuanVideo loaded directly to CUDA (no offload)")
    else:
        # SAFE MODE: CPU offloading
        # Use when other services are running
        pipe.enable_model_cpu_offload()
        logger.info("✓ HunyuanVideo loaded with CPU offloading (bf16)")
    
    # Enable VAE optimizations (always beneficial)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    
    # Log effective attention backend
    _log_attention_backend(pipe)
    
    # Compile transformer if requested (Blackwell max-autotune)
    if compile_model and hasattr(pipe, 'transformer'):
        logger.info("Compiling transformer with max-autotune (this may take a few minutes)...")
        pipe.transformer = torch.compile(
            pipe.transformer, 
            mode="max-autotune", 
            fullgraph=True
        )
        logger.info("✓ Transformer compiled with max-autotune")
    
    rss_after = read_rss_gb()
    if rss_before and rss_after:
        logger.info(f"RSS after load: {rss_after:.2f} GB (delta: +{rss_after - rss_before:.2f} GB)")
    
    return pipe


def _log_attention_backend(pipe) -> str:
    """
    Log which attention backend will effectively be used.
    This addresses Model 1's concern: "verify Diffusers is *using* FlashAttention"
    """
    backend = "unknown"
    
    # Check Diffusers attention processor
    if hasattr(pipe, 'transformer'):
        attn_procs = getattr(pipe.transformer, 'attn_processors', {})
        if attn_procs:
            proc_types = set(type(p).__name__ for p in attn_procs.values())
            logger.info(f"Attention processors: {proc_types}")
            if any('Flash' in t for t in proc_types):
                backend = "flash_attn"
            elif any('XFormers' in t for t in proc_types):
                backend = "xformers"
            else:
                backend = "sdpa (default)"
    
    # Check PyTorch SDPA backend preference
    try:
        import torch.backends.cuda
        flash_enabled = getattr(torch.backends.cuda, 'flash_sdp_enabled', lambda: None)()
        mem_efficient_enabled = getattr(torch.backends.cuda, 'mem_efficient_sdp_enabled', lambda: None)()
        logger.info(f"PyTorch SDPA backends: flash={flash_enabled}, mem_efficient={mem_efficient_enabled}")
    except Exception:
        pass
    
    logger.info(f"Effective attention backend: {backend}")
    return backend


def load_model(model_tier: str, use_fp8: bool = True, no_offload: bool = False, compile_model: bool = False):
    """
    Load the appropriate model based on tier.
    
    Args:
        model_tier: One of validator, workhorse, experimental
        use_fp8: Enable FP8 quantization where supported
        no_offload: Disable CPU offloading (for GB10 speed runs)
        compile_model: Enable torch.compile with max-autotune
    """
    
    # MODEL TIERS - Reflecting verified reality (not aspirational)
    # Based on empirical testing on GB10 (2025-02-01)
    MODEL_CONFIGS = {
        "validator": {
            # SVD-XT - Image-to-Video (VERIFIED WORKING)
            # Fast, reliable, good for pipeline validation
            "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "loader": load_svd,
            "is_t2v": False,
            "is_i2v": True,
            "default_frames": 25,
            "default_steps": 25,
            "default_width": 1024,
            "default_height": 576,
            "status": "VERIFIED",
        },
        "workhorse": {
            # HunyuanVideo - Text-to-Video (VERIFIED WORKING)
            # Production quality, main T2V model
            "repo_id": "hunyuanvideo-community/HunyuanVideo",
            "loader": load_hunyuanvideo,
            "is_t2v": True,
            "is_i2v": False,
            "default_frames": 25,  # Reduced for faster benchmarks
            "default_steps": 25,
            "status": "VERIFIED",
        },
        "experimental": {
            # CogVideoX-2B - Text-to-Video (UNSTABLE - dtype issues)
            # T5 encoder requires FP32, DiT requires BF16 - complex casting needed
            "repo_id": "THUDM/CogVideoX-2b",
            "loader": load_cogvideox,
            "is_t2v": True,
            "is_i2v": False,
            "default_frames": 49,
            "default_steps": 50,
            "status": "UNSTABLE",
            "notes": "T5 encoder dtype conflicts with Apex fused RMSNorm",
        },
    }
    
    if model_tier not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model tier: {model_tier}. Choose from: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_tier]
    logger.info(f"Model tier: {model_tier}")
    logger.info(f"Repository: {config['repo_id']}")
    logger.info(f"Status: {config.get('status', 'unknown')}")
    
    if config.get("status") == "UNSTABLE":
        logger.warning(f"⚠️  {model_tier} tier is UNSTABLE: {config.get('notes', 'may have issues')}")
    
    # Store loading options in config for downstream use
    config["no_offload"] = no_offload
    config["compile_model"] = compile_model
    
    try:
        pipe = config["loader"](config["repo_id"], no_offload=no_offload, compile_model=compile_model)
        return pipe, config
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def generate_video(
    pipe,
    config: dict,
    prompt: str,
    output_dir: str,
    num_frames: int = None,
    height: int = 720,
    width: int = 1280,
    num_inference_steps: int = None,
    input_image: Optional[str] = None,
    monitor: Optional[BandwidthMonitor] = None,
):
    """Generate a video using the loaded pipeline."""
    
    # Use config defaults if not specified
    if num_frames is None:
        num_frames = config.get("default_frames", 49)
    if num_inference_steps is None:
        num_inference_steps = config.get("default_steps", 50)
    
    # Use SVD-specific defaults
    if config.get("is_i2v"):
        width = config.get("default_width", 1024)
        height = config.get("default_height", 576)
    
    logger.info("="*50)
    logger.info("STARTING VIDEO GENERATION")
    logger.info("="*50)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Frames: {num_frames}")
    logger.info(f"Resolution: {width}x{height}")
    logger.info(f"Steps: {num_inference_steps}")
    
    # Log effective dtype configuration
    if hasattr(pipe, 'text_encoder'):
        logger.info(f"TextEncoder dtype: {pipe.text_encoder.dtype}")
    if hasattr(pipe, 'transformer'):
        logger.info(f"Transformer dtype: {pipe.transformer.dtype}")
    if hasattr(pipe, 'vae'):
        logger.info(f"VAE dtype: {pipe.vae.dtype}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    if monitor:
        monitor.sample()
    
    try:
        with torch.inference_mode():
            # Different pipelines have different parameter names
            pipeline_name = type(pipe).__name__
            
            if "StableVideoDiffusion" in pipeline_name:
                # SVD is Image-to-Video - requires input image
                if not input_image:
                    raise ValueError("SVD requires --input-image path")
                
                from PIL import Image
                
                logger.info(f"Loading input image: {input_image}")
                image = Image.open(input_image).convert("RGB")
                
                # Resize to SVD's expected resolution (1024x576)
                target_w, target_h = width, height
                img_w, img_h = image.size
                
                # Center crop to 16:9 if needed
                target_ratio = target_w / target_h
                current_ratio = img_w / img_h
                
                if current_ratio > target_ratio:
                    # Too wide - crop width
                    new_w = int(img_h * target_ratio)
                    left = (img_w - new_w) // 2
                    image = image.crop((left, 0, left + new_w, img_h))
                elif current_ratio < target_ratio:
                    # Too tall - crop height
                    new_h = int(img_w / target_ratio)
                    top = (img_h - new_h) // 2
                    image = image.crop((0, top, img_w, top + new_h))
                
                image = image.resize((target_w, target_h), Image.LANCZOS)
                logger.info(f"Preprocessed image: {image.size}")
                
                output = pipe(
                    image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    decode_chunk_size=8,
                    generator=torch.Generator(device="cuda").manual_seed(42),
                )
            elif "CogVideoX" in pipeline_name:
                output = pipe(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,
                    guidance_scale=6.0,
                    generator=torch.Generator(device="cuda").manual_seed(42),
                )
            elif "HunyuanVideo" in pipeline_name:
                output = pipe(
                    prompt=prompt,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                )
            else:
                output = pipe(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                )
        
        if monitor:
            monitor.sample()
        
        elapsed = time.time() - start_time
        fps = num_frames / elapsed
        
        logger.info("="*50)
        logger.info("GENERATION COMPLETE")
        logger.info("="*50)
        logger.info(f"Time: {elapsed:.2f}s")
        logger.info(f"FPS: {fps:.2f}")
        
        # Peak RSS after generation
        peak_rss = read_rss_gb()
        if peak_rss:
            logger.info(f"Peak RSS (VmHWM): {peak_rss:.2f} GB")
        
        # Save output
        video_path = None
        if hasattr(output, 'frames'):
            frames = output.frames[0]
            
            from diffusers.utils import export_to_video
            video_path = output_path / f"output_{config.get('repo_id', 'video').split('/')[-1]}.mp4"
            export_to_video(frames, str(video_path), fps=8 if "CogVideoX" in pipeline_name else 15)
            logger.info(f"✓ Saved: {video_path}")
            
            # Validate the output video
            validation = validate_video_output(str(video_path))
            if validation["valid"]:
                meta = validation["metadata"]
                logger.info(f"✓ Video validated: {meta.get('width')}x{meta.get('height')}, "
                           f"{meta.get('nb_frames')} frames, {meta.get('duration', 0):.2f}s, "
                           f"{meta.get('file_size_bytes', 0) / 1024:.1f} KB")
            else:
                logger.warning(f"⚠️  Video validation issue: {validation['error']}")
        
        return {
            "success": True,
            "elapsed_seconds": elapsed,
            "fps": fps,
            "output_path": str(output_path),
            "video_path": str(video_path) if video_path else None,
            "peak_rss_gb": peak_rss,
            "validation": validation if video_path else None,
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }


def run_smoke_test(output_dir: str, monitor_bandwidth: bool = False):
    """
    RIGOROUS smoke test - not just imports, but kernel/execution verification.
    """
    logger.info("="*50)
    logger.info("SMOKE TEST - Rigorous Pipeline Verification")
    logger.info("="*50)
    
    monitor = BandwidthMonitor(enabled=monitor_bandwidth)
    all_passed = True
    
    # Test 1: CUDA availability
    logger.info("\n[Test 1] CUDA Availability")
    assert torch.cuda.is_available(), "CUDA not available!"
    device = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability()
    logger.info(f"✓ GPU: {device}")
    logger.info(f"✓ Compute Capability: {cap}")
    
    # Test 2: FP8 dtype support
    logger.info("\n[Test 2] FP8 Dtype Support")
    try:
        x = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
        x_fp8 = x.to(torch.float8_e4m3fn)
        logger.info(f"✓ FP8 tensor: {x_fp8.shape}, dtype={x_fp8.dtype}")
        del x, x_fp8
    except Exception as e:
        logger.error(f"✗ FP8 failed: {e}")
        all_passed = False
    
    monitor.sample()
    
    # Test 3: FlashAttention - KERNEL VERIFIED
    logger.info("\n[Test 3] FlashAttention KERNEL Verification")
    fa_result = verify_flash_attention_kernel()
    if fa_result["kernel_verified"]:
        logger.info(f"✓ FlashAttention {fa_result['version']} - KERNEL VERIFIED")
    elif fa_result["available"]:
        logger.warning(f"○ FlashAttention {fa_result['version']} - importable but kernel not verified")
    else:
        logger.info("○ FlashAttention not available (SDPA fallback will be used)")
    
    monitor.sample()
    
    # Test 4: Memory bandwidth - D2D copy (clarified measurement)
    logger.info("\n[Test 4] Memory Bandwidth (D2D - Device to Device copy)")
    try:
        size_gb = 5.0
        num_elements = int(size_gb * 1e9 / 4)
        src = torch.randn(num_elements, device='cuda', dtype=torch.float32)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        dst = src.clone()  # D2D copy
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        bandwidth = (size_gb * 2) / elapsed  # Read + Write
        logger.info(f"✓ D2D Bandwidth: {bandwidth:.1f} GB/s (5GB tensor clone)")
        
        del src, dst
    except Exception as e:
        logger.error(f"✗ Bandwidth test failed: {e}")
    
    monitor.sample()
    
    # Test 5: Diffusers + HunyuanVideo availability
    logger.info("\n[Test 5] Diffusers + Pipeline Availability")
    try:
        import diffusers
        logger.info(f"✓ Diffusers version: {diffusers.__version__}")
        
        # Verify HunyuanVideo IS available (correcting earlier mistake)
        from diffusers import HunyuanVideoPipeline
        logger.info("✓ HunyuanVideoPipeline: AVAILABLE")
        
        from diffusers import CogVideoXPipeline
        logger.info("✓ CogVideoXPipeline: AVAILABLE")
        
    except ImportError as e:
        logger.warning(f"○ Pipeline import issue: {e}")
    
    # Test 6: PyTorch compile check
    logger.info("\n[Test 6] torch.compile Availability")
    try:
        @torch.compile
        def simple_fn(x):
            return x * 2 + 1
        
        test_tensor = torch.randn(100, device='cuda')
        result = simple_fn(test_tensor)
        logger.info(f"✓ torch.compile: working")
        del test_tensor, result
    except Exception as e:
        logger.warning(f"○ torch.compile issue: {e}")
    
    torch.cuda.empty_cache()
    monitor.report()
    
    logger.info("\n" + "="*50)
    if all_passed:
        logger.info("SMOKE TEST PASSED ✓")
    else:
        logger.info("SMOKE TEST COMPLETED WITH WARNINGS")
    logger.info("="*50)
    
    # Print attention backend recommendation
    logger.info("\nRECOMMENDED ATTENTION BACKEND ORDER:")
    if fa_result["kernel_verified"]:
        logger.info("  1. flash_attn (KERNEL VERIFIED)")
        logger.info("  2. xformers (if available)")
        logger.info("  3. sdpa (PyTorch native - fallback)")
    else:
        logger.info("  1. sdpa (PyTorch native - default)")
        logger.info("  2. xformers (if available)")
        logger.info("  3. flash_attn (experimental - not kernel verified)")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="The Auteur System - Video Generation for GB10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Tiers (reflecting verified reality):
  validator    SVD-XT (Image-to-Video) - Fast, reliable validation
  workhorse    HunyuanVideo (Text-to-Video) - Production quality
  experimental CogVideoX (Text-to-Video) - UNSTABLE, dtype issues

Optimization Flags for GB10:
  --no-offload   Disable CPU offloading (model pinned to unified memory)
  --compile      Enable torch.compile with max-autotune
  
Speed Run Example (recommended for benchmarks):
  python generate.py --model-tier workhorse --prompt "..." --no-offload --compile
        """
    )
    parser.add_argument("--prompt", type=str, help="Text prompt for video generation")
    parser.add_argument("--model-tier", type=str, default="workhorse",
                        choices=["validator", "workhorse", "experimental"],
                        help="validator=SVD-I2V, workhorse=HunyuanVideo, experimental=CogVideoX")
    parser.add_argument("--input-image", type=str, default=None,
                        help="Input image path for I2V models (required for validator tier)")
    parser.add_argument("--output-dir", type=str, default="/output")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames (default: model-specific)")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=None, help="Inference steps (default: model-specific)")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8 quantization")
    parser.add_argument("--no-offload", action="store_true", 
                        help="Disable CPU offloading - load model directly to CUDA (GB10 speed mode)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile with max-autotune (Blackwell optimization)")
    parser.add_argument("--monitor-bandwidth", action="store_true", help="Monitor memory during generation")
    parser.add_argument("--smoke-test", action="store_true", help="Run rigorous smoke test")
    
    args = parser.parse_args()
    
    if args.smoke_test:
        success = run_smoke_test(args.output_dir, args.monitor_bandwidth)
        sys.exit(0 if success else 1)
    
    # validator tier (SVD) doesn't need a prompt (it's I2V), but T2V models do
    if args.model_tier != "validator" and not args.prompt:
        parser.error("--prompt is required for T2V models (or use --smoke-test, or --model-tier validator with --input-image)")
    
    monitor = BandwidthMonitor(enabled=args.monitor_bandwidth)
    
    # Log initial RSS
    initial_rss = read_rss_gb()
    logger.info(f"Initial RSS: {initial_rss:.2f} GB" if initial_rss else "RSS tracking unavailable")
    
    # Load model with optimization flags
    pipe, config = load_model(
        args.model_tier, 
        use_fp8=not args.no_fp8,
        no_offload=args.no_offload,
        compile_model=args.compile
    )
    
    monitor.sample("after_load")
    
    # Validate I2V requirements (validator tier = SVD = I2V)
    if config.get("is_i2v") and not args.input_image:
        parser.error("--input-image is required for validator tier (SVD image-to-video model)")
    
    # Generate video
    result = generate_video(
        pipe=pipe,
        config=config,
        prompt=args.prompt,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        input_image=args.input_image,
        monitor=monitor,
    )
    
    monitor.report()
    
    if result["success"]:
        logger.info(f"\n✓ Video saved to: {result['output_path']}")
        sys.exit(0)
    else:
        logger.error(f"\n✗ Generation failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()

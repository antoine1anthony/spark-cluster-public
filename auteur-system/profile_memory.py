#!/usr/bin/env python3
"""
Memory Profiler for The Auteur System
=====================================

This script loads HunyuanVideo layer-by-layer and reports exact VRAM usage 
per component. This tells us definitively why a ~20GB model demands offloading 
on an 80GB-free system.

Key insight: On GB10's unified memory, "CPU" and "GPU" memory are the same
physical LPDDR5X. The offload penalty is software overhead, not physical.
"""

import gc
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='[Profile] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory state at a point in time"""
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    
    def __str__(self):
        return f"Alloc: {self.allocated_gb:.2f}GB | Reserved: {self.reserved_gb:.2f}GB | Free: {self.free_gb:.2f}GB"


def get_memory_snapshot() -> MemorySnapshot:
    """Get current GPU memory state"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    
    # Get total memory
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - reserved
    
    return MemorySnapshot(allocated, reserved, free)


def clear_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def profile_model_component(name: str, load_fn, baseline: MemorySnapshot) -> Dict[str, Any]:
    """Load a component and measure its memory impact"""
    logger.info(f"Loading: {name}...")
    
    try:
        component = load_fn()
        current = get_memory_snapshot()
        
        delta_alloc = current.allocated_gb - baseline.allocated_gb
        delta_reserved = current.reserved_gb - baseline.reserved_gb
        
        # Get dtype info if available
        dtype_info = "N/A"
        if hasattr(component, 'dtype'):
            dtype_info = str(component.dtype)
        elif hasattr(component, 'parameters'):
            try:
                first_param = next(component.parameters())
                dtype_info = str(first_param.dtype)
            except StopIteration:
                pass
        
        result = {
            "name": name,
            "allocated_gb": delta_alloc,
            "reserved_gb": delta_reserved,
            "dtype": dtype_info,
            "current_total": current.allocated_gb,
            "success": True
        }
        
        logger.info(f"  ✓ {name}: +{delta_alloc:.2f}GB allocated (dtype: {dtype_info})")
        
        return result, component, current
        
    except Exception as e:
        logger.error(f"  ✗ {name} failed: {e}")
        return {"name": name, "error": str(e), "success": False}, None, baseline


def profile_hunyuanvideo():
    """
    Profile HunyuanVideo component-by-component
    """
    from diffusers import HunyuanVideoPipeline
    from diffusers.models import HunyuanVideoTransformer3DModel
    from transformers import LlamaModel, CLIPTextModel
    
    logger.info("=" * 60)
    logger.info("HUNYUANVIDEO MEMORY PROFILE")
    logger.info("=" * 60)
    
    clear_memory()
    baseline = get_memory_snapshot()
    logger.info(f"Baseline: {baseline}")
    
    results = []
    components = {}
    
    repo_id = "hunyuanvideo-community/HunyuanVideo"
    
    # Profile each component separately
    logger.info("\n--- Loading Components Individually ---\n")
    
    # 1. Text Encoder (CLIP)
    def load_text_encoder():
        from transformers import CLIPTextModel
        return CLIPTextModel.from_pretrained(
            repo_id, 
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16
        ).to("cuda")
    
    result, comp, current = profile_model_component("text_encoder (CLIP)", load_text_encoder, baseline)
    results.append(result)
    if comp:
        components["text_encoder"] = comp
        baseline = current
    
    # 2. Text Encoder 2 (LLaMA)
    def load_text_encoder_2():
        from transformers import LlamaModel
        return LlamaModel.from_pretrained(
            repo_id,
            subfolder="text_encoder_2", 
            torch_dtype=torch.bfloat16
        ).to("cuda")
    
    result, comp, current = profile_model_component("text_encoder_2 (LLaMA)", load_text_encoder_2, baseline)
    results.append(result)
    if comp:
        components["text_encoder_2"] = comp
        baseline = current
    
    # 3. Transformer (DiT)
    def load_transformer():
        return HunyuanVideoTransformer3DModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        ).to("cuda")
    
    result, comp, current = profile_model_component("transformer (DiT)", load_transformer, baseline)
    results.append(result)
    if comp:
        components["transformer"] = comp
        baseline = current
    
    # 4. VAE
    def load_vae():
        from diffusers import AutoencoderKLHunyuanVideo
        return AutoencoderKLHunyuanVideo.from_pretrained(
            repo_id,
            subfolder="vae",
            torch_dtype=torch.bfloat16
        ).to("cuda")
    
    result, comp, current = profile_model_component("vae", load_vae, baseline)
    results.append(result)
    if comp:
        components["vae"] = comp
        baseline = current
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MEMORY PROFILE SUMMARY")
    logger.info("=" * 60)
    
    total_alloc = sum(r.get("allocated_gb", 0) for r in results if r.get("success"))
    
    for r in results:
        if r.get("success"):
            pct = (r["allocated_gb"] / total_alloc * 100) if total_alloc > 0 else 0
            logger.info(f"  {r['name']:30} {r['allocated_gb']:6.2f} GB  ({pct:5.1f}%)  [{r['dtype']}]")
        else:
            logger.info(f"  {r['name']:30} FAILED: {r.get('error', 'unknown')}")
    
    logger.info("-" * 60)
    logger.info(f"  {'TOTAL':30} {total_alloc:6.2f} GB")
    
    final = get_memory_snapshot()
    logger.info(f"\nFinal Memory State: {final}")
    
    # Cleanup
    logger.info("\n--- Cleanup Test ---")
    for name, comp in components.items():
        del comp
    components.clear()
    clear_memory()
    
    after_cleanup = get_memory_snapshot()
    logger.info(f"After cleanup: {after_cleanup}")
    
    return results


def profile_with_fp8():
    """
    Test FP8 quantization impact on memory
    """
    logger.info("\n" + "=" * 60)
    logger.info("FP8 QUANTIZATION TEST")
    logger.info("=" * 60)
    
    clear_memory()
    baseline = get_memory_snapshot()
    
    # Create a test tensor to simulate model weights
    size_gb = 5.0
    num_elements = int(size_gb * 1e9 / 2)  # BF16 = 2 bytes
    
    logger.info(f"\nCreating {size_gb}GB test tensor in BF16...")
    tensor_bf16 = torch.randn(num_elements, device='cuda', dtype=torch.bfloat16)
    after_bf16 = get_memory_snapshot()
    logger.info(f"  BF16 tensor: {after_bf16.allocated_gb - baseline.allocated_gb:.2f}GB")
    
    # Convert to FP8
    logger.info("Converting to FP8...")
    try:
        tensor_fp8 = tensor_bf16.to(torch.float8_e4m3fn)
        after_fp8 = get_memory_snapshot()
        
        # Delete BF16 to see FP8 alone
        del tensor_bf16
        clear_memory()
        fp8_only = get_memory_snapshot()
        
        logger.info(f"  FP8 tensor alone: {fp8_only.allocated_gb - baseline.allocated_gb:.2f}GB")
        logger.info(f"  Memory savings: {(1 - (fp8_only.allocated_gb - baseline.allocated_gb) / (after_bf16.allocated_gb - baseline.allocated_gb)) * 100:.1f}%")
        
        del tensor_fp8
    except Exception as e:
        logger.error(f"FP8 conversion failed: {e}")
    
    clear_memory()


def profile_attention_memory():
    """
    Profile FlashAttention vs SDPA memory usage
    """
    logger.info("\n" + "=" * 60)
    logger.info("ATTENTION MEMORY PROFILE")
    logger.info("=" * 60)
    
    clear_memory()
    
    # Test parameters (realistic for video generation)
    batch_size = 1
    seq_len = 4096  # Typical for video latents
    num_heads = 24
    head_dim = 64
    
    logger.info(f"Test config: B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}")
    
    # Create Q, K, V
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    
    baseline = get_memory_snapshot()
    logger.info(f"QKV allocated: {baseline}")
    
    # Test FlashAttention
    logger.info("\n1. FlashAttention-2:")
    try:
        from flash_attn import flash_attn_func
        clear_memory()
        torch.cuda.reset_peak_memory_stats()
        
        out_fa = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        torch.cuda.synchronize()
        
        peak_fa = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(f"  Peak memory: {peak_fa:.2f}GB")
        del out_fa
    except Exception as e:
        logger.error(f"  FlashAttention failed: {e}")
    
    # Test SDPA
    logger.info("\n2. PyTorch SDPA:")
    clear_memory()
    torch.cuda.reset_peak_memory_stats()
    
    # SDPA expects (B, H, S, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
    torch.cuda.synchronize()
    
    peak_sdpa = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info(f"  Peak memory: {peak_sdpa:.2f}GB")
    
    del q, k, v, q_sdpa, k_sdpa, v_sdpa, out_sdpa
    clear_memory()


def main():
    """Run all memory profiles"""
    logger.info("=" * 60)
    logger.info("THE AUTEUR SYSTEM - MEMORY PROFILER")
    logger.info("=" * 60)
    
    # System info
    logger.info(f"\nGPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    
    initial = get_memory_snapshot()
    logger.info(f"Initial state: {initial}")
    
    # Run profiles
    profile_with_fp8()
    profile_attention_memory()
    
    # Main event: HunyuanVideo profile
    logger.info("\n" + "=" * 60)
    logger.info("MAIN PROFILE: HUNYUANVIDEO")
    logger.info("=" * 60)
    
    try:
        results = profile_hunyuanvideo()
        
        # Recommendations
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION RECOMMENDATIONS")
        logger.info("=" * 60)
        
        total = sum(r.get("allocated_gb", 0) for r in results if r.get("success"))
        
        if total > 60:
            logger.info("⚠️  Model exceeds 60GB - CPU offload required")
            logger.info("   Consider: FP8 quantization for transformer")
        elif total > 40:
            logger.info("⚠️  Model uses 40-60GB - tight fit")
            logger.info("   Consider: VAE tiling, smaller batch")
        else:
            logger.info("✓  Model fits comfortably in memory")
            logger.info("   Try: disable CPU offload, use torch.compile")
        
        # Find the biggest component
        biggest = max(results, key=lambda x: x.get("allocated_gb", 0))
        logger.info(f"\n💡 Biggest component: {biggest['name']} ({biggest.get('allocated_gb', 0):.2f}GB)")
        logger.info("   This is your optimization target!")
        
    except Exception as e:
        logger.error(f"HunyuanVideo profile failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

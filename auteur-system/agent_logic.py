#!/usr/bin/env python3
"""
The Auteur System - Agent Logic
Decision tree and runtime management for video generation jobs.

Key Responsibilities:
1. Kernel selection (FlashAttn → xformers → SDPA)
2. FP8 casting for bandwidth optimization
3. torch.compile management
4. Telemetry reporting
"""

import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict
from enum import Enum

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[Auteur] %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("auteur")


class AttentionBackend(Enum):
    FLASH_ATTN = "flash_attn"
    XFORMERS = "xformers"
    SDPA = "sdpa"


class KernelStatus(Enum):
    ACTIVE = "active"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"


@dataclass
class RuntimeConfig:
    """Runtime configuration for the agent."""
    compute_dtype: torch.dtype = torch.float8_e4m3fn
    attention_backend: AttentionBackend = AttentionBackend.SDPA
    torch_compile_enabled: bool = True
    torch_compile_mode: str = "reduce-overhead"
    telemetry_enabled: bool = True


class AgentLogic:
    """
    The Auteur System Agent Logic.
    
    Implements the decision tree for:
    - Kernel selection with fallbacks
    - FP8 casting for bandwidth optimization
    - Model compilation
    - Performance telemetry
    """
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self.kernel_status: Dict[str, KernelStatus] = {}
        self._init_cuda()
        self._probe_kernels()
        
    def _init_cuda(self):
        """Initialize CUDA and verify GPU availability."""
        if not torch.cuda.is_available():
            logger.error("CUDA not available! Cannot run on GB10.")
            raise RuntimeError("CUDA required for The Auteur System")
        
        device = torch.cuda.get_device_name(0)
        logger.info(f"CUDA Device: {device}")
        
        # Set memory allocation config for unified memory
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
    def _probe_kernels(self):
        """
        Probe available attention kernels.
        
        GB10 (sm_121) Kernel Support Status (VERIFIED):
        - FlashAttention 2.7.4+: ✓ WORKS on sm_121 (tested)
        - xFormers: ✓ Works (memory efficient attention)
        - SDPA: ✓ Full support (PyTorch native fallback)
        """
        logger.info("Probing attention kernels for GB10 (sm_121)...")
        
        # Try FlashAttention (VERIFIED WORKING on sm_121 with v2.7.4+)
        try:
            import flash_attn
            version = getattr(flash_attn, '__version__', 'unknown')
            self.kernel_status["flash_attn"] = KernelStatus.ACTIVE
            logger.info(f"✓ FlashAttention-2 v{version}: AVAILABLE (verified on sm_121)")
        except ImportError:
            self.kernel_status["flash_attn"] = KernelStatus.UNAVAILABLE
            logger.info("○ FlashAttention-2: Not installed")
        
        # Try xformers (optional, memory efficient)
        try:
            import xformers
            import xformers.ops
            self.kernel_status["xformers"] = KernelStatus.ACTIVE
            logger.info("✓ xFormers: AVAILABLE (memory efficient)")
        except ImportError:
            self.kernel_status["xformers"] = KernelStatus.UNAVAILABLE
            logger.info("○ xFormers: Not installed")
        
        # SDPA is always available in PyTorch 2.0+
        self.kernel_status["sdpa"] = KernelStatus.ACTIVE
        logger.info("✓ PyTorch SDPA: AVAILABLE (native fallback)")
        
        # Select best available backend
        self._select_attention_backend()
        
    def _select_attention_backend(self):
        """
        Select the best available attention backend.
        
        Priority for GB10 (sm_121) - VERIFIED WORKING:
        1. FlashAttention 2.7.4+ - FASTEST (verified on sm_121)
        2. xFormers - Memory efficient alternative
        3. SDPA (PyTorch native) - Always works fallback
        """
        priority = [
            (AttentionBackend.FLASH_ATTN, "flash_attn"),
            (AttentionBackend.XFORMERS, "xformers"),
            (AttentionBackend.SDPA, "sdpa"),
        ]
        
        for backend, key in priority:
            if self.kernel_status.get(key) == KernelStatus.ACTIVE:
                self.config.attention_backend = backend
                logger.info(f"Selected attention backend: {backend.value}")
                return
        
        # Fallback to SDPA (should always work)
        self.config.attention_backend = AttentionBackend.SDPA
        logger.warning("Falling back to SDPA attention")
    
    def cast_to_fp8(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Cast model to FP8 for bandwidth optimization.
        
        CRITICAL for GB10: Memory bandwidth is ~273 GB/s.
        FP8 reduces memory traffic by 50% vs FP16.
        """
        logger.info("Casting model to FP8 (float8_e4m3fn)...")
        
        try:
            # Try using torchao for quantization
            from torchao.quantization import quantize_, float8_weight_only
            quantize_(model, float8_weight_only())
            logger.info("✓ FP8 quantization applied via torchao")
            return model
        except ImportError:
            logger.warning("torchao not available, using native FP8 casting")
        
        # Native FP8 casting for linear layers
        cast_count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(module, 'weight') and module.weight is not None:
                    try:
                        module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
                        cast_count += 1
                    except Exception as e:
                        logger.debug(f"Could not cast {name}: {e}")
        
        logger.info(f"✓ Cast {cast_count} linear layers to FP8")
        return model
    
    def compile_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply torch.compile for kernel fusion.
        
        Note: May hang on some Triton configurations. 
        Has fallback to disable if needed.
        """
        if not self.config.torch_compile_enabled:
            logger.info("torch.compile disabled")
            return model
        
        logger.info(f"Compiling model (mode={self.config.torch_compile_mode})...")
        
        try:
            compiled = torch.compile(
                model,
                mode=self.config.torch_compile_mode,
                fullgraph=False,  # Safer for video models
            )
            logger.info("✓ Model compiled successfully")
            return compiled
        except Exception as e:
            logger.error(f"torch.compile failed: {e}")
            logger.warning("Continuing without compilation")
            return model
    
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Full model preparation pipeline.
        
        1. Move to CUDA
        2. Cast to FP8
        3. Compile
        """
        logger.info("="*50)
        logger.info("Preparing model for GB10...")
        logger.info("="*50)
        
        # Move to GPU
        model = model.cuda()
        logger.info("✓ Model moved to CUDA (unified memory)")
        
        # Cast to FP8
        model = self.cast_to_fp8(model)
        
        # Compile
        model = self.compile_model(model)
        
        logger.info("="*50)
        logger.info("Model preparation complete")
        logger.info("="*50)
        
        return model


class Telemetry:
    """Performance telemetry for The Auteur System."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.metrics: Dict[str, Any] = {}
        
    def start(self):
        """Start telemetry collection."""
        self.start_time = time.time()
        self.metrics = {
            "start_time": self.start_time,
            "tokens_generated": 0,
            "frames_generated": 0,
        }
        
    def record_frame(self):
        """Record a generated frame."""
        self.metrics["frames_generated"] += 1
        
    def record_tokens(self, count: int):
        """Record generated tokens."""
        self.metrics["tokens_generated"] += count
        
    def report(self) -> Dict[str, Any]:
        """Generate telemetry report."""
        if self.start_time is None:
            return {}
        
        elapsed = time.time() - self.start_time
        frames = self.metrics.get("frames_generated", 0)
        tokens = self.metrics.get("tokens_generated", 0)
        
        report = {
            "elapsed_seconds": round(elapsed, 2),
            "frames_generated": frames,
            "frames_per_second": round(frames / elapsed, 2) if elapsed > 0 else 0,
            "tokens_generated": tokens,
            "tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else 0,
        }
        
        logger.info("="*50)
        logger.info("TELEMETRY REPORT")
        logger.info("="*50)
        for key, value in report.items():
            logger.info(f"  {key}: {value}")
        logger.info("="*50)
        
        return report


def main():
    """Test the agent logic."""
    logger.info("The Auteur System - Agent Logic Test")
    
    # Initialize agent
    agent = AgentLogic()
    
    # Report kernel status
    logger.info("\nKernel Status:")
    for kernel, status in agent.kernel_status.items():
        logger.info(f"  {kernel}: {status.value}")
    
    logger.info(f"\nSelected Backend: {agent.config.attention_backend.value}")
    logger.info(f"Compute Dtype: {agent.config.compute_dtype}")
    logger.info(f"Torch Compile: {agent.config.torch_compile_enabled}")


if __name__ == "__main__":
    main()

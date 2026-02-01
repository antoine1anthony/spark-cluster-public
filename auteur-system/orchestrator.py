#!/usr/bin/env python3
"""
The Auteur System - Orchestrator
Manages job distribution across the DGX Spark cluster.

Execution Modes:
- Factory (Mode A): Independent workers, 2x throughput
- Titan (Mode B): Ring Sequence Parallelism for large models
"""

import os
import yaml
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


class ExecutionMode(Enum):
    FACTORY = "factory"  # Independent workers (default)
    TITAN = "titan"      # Distributed for large models


@dataclass
class ClusterNode:
    name: str
    ip: str
    role: str
    gpu_memory_gb: int
    available: bool = True


@dataclass
class VideoJob:
    job_id: str
    prompt: str
    model_tier: str  # validator, workhorse, heavy
    num_frames: int = 49
    height: int = 720
    width: int = 1280
    priority: int = 0


class Orchestrator:
    """
    The Auteur System Orchestrator.
    
    Responsibilities:
    - Load cluster configuration
    - Determine execution mode based on model size
    - Dispatch jobs to appropriate nodes
    - Monitor job status
    """
    
    def __init__(self, config_path: str = "/app/configs/inference_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.nodes = self._init_nodes()
        self.model_size_threshold = 110  # GB
        
    def _load_config(self) -> Dict[str, Any]:
        """Load inference configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if file not found."""
        return {
            "cluster": {
                "spark3": {"ip": os.environ.get("SPARK3_HOST", "localhost"), "role": "primary", "gpu_memory_gb": 128},
                "spark4": {"ip": os.environ.get("SPARK4_HOST", "localhost"), "role": "secondary", "gpu_memory_gb": 128},
            },
            "execution": {
                "factory": {"model_size_limit_gb": 110},
            },
            "models": {
                "validator": {"memory_estimate_gb": 8},
                "workhorse": {"memory_estimate_gb": 45},
                "heavy": {"memory_estimate_gb": 95},
            }
        }
    
    def _init_nodes(self) -> List[ClusterNode]:
        """Initialize cluster nodes from config."""
        nodes = []
        for name, cfg in self.config.get("cluster", {}).items():
            nodes.append(ClusterNode(
                name=name,
                ip=cfg.get("ip", ""),
                role=cfg.get("role", "worker"),
                gpu_memory_gb=cfg.get("gpu_memory_gb", 128),
            ))
        return nodes
    
    def get_model_size(self, model_tier: str) -> float:
        """Get estimated model size in GB."""
        models = self.config.get("models", {})
        model_cfg = models.get(model_tier, {})
        return model_cfg.get("memory_estimate_gb", 50)
    
    def determine_mode(self, job: VideoJob) -> ExecutionMode:
        """
        Determine execution mode based on model size.
        
        - Factory (Mode A): model_size < 110GB → Single node, independent
        - Titan (Mode B): model_size >= 110GB → Distributed ring parallel
        """
        model_size = self.get_model_size(job.model_tier)
        
        if model_size >= self.model_size_threshold:
            print(f"[Orchestrator] Model size ({model_size}GB) >= threshold. Using TITAN mode.")
            return ExecutionMode.TITAN
        else:
            print(f"[Orchestrator] Model size ({model_size}GB) < threshold. Using FACTORY mode.")
            return ExecutionMode.FACTORY
    
    def select_node(self, mode: ExecutionMode) -> ClusterNode:
        """Select the best node for job execution."""
        available_nodes = [n for n in self.nodes if n.available]
        
        if not available_nodes:
            raise RuntimeError("No available nodes in cluster")
        
        # For Factory mode, prefer primary node
        if mode == ExecutionMode.FACTORY:
            primary = [n for n in available_nodes if n.role == "primary"]
            return primary[0] if primary else available_nodes[0]
        
        # For Titan mode, return primary (will coordinate with secondary)
        return available_nodes[0]
    
    def dispatch_factory(self, job: VideoJob, node: ClusterNode) -> str:
        """Dispatch job in Factory mode (single node)."""
        print(f"[Factory] Dispatching job {job.job_id} to {node.name} ({node.ip})")
        
        cmd = [
            "python", "/app/generate.py",
            "--prompt", job.prompt,
            "--model-tier", job.model_tier,
            "--num-frames", str(job.num_frames),
            "--height", str(job.height),
            "--width", str(job.width),
            "--output-dir", f"/app/outputs/{job.job_id}",
        ]
        
        return " ".join(cmd)
    
    def dispatch_titan(self, job: VideoJob) -> str:
        """Dispatch job in Titan mode (distributed)."""
        print(f"[Titan] Dispatching distributed job {job.job_id}")
        
        # Get all nodes
        primary = [n for n in self.nodes if n.role == "primary"][0]
        
        cmd = [
            "torchrun",
            "--nproc_per_node=1",
            "--nnodes=2",
            "--node_rank=0",
            f"--master_addr={primary.ip}",
            "/app/generate.py",
            "--prompt", f'"{job.prompt}"',
            "--model-tier", job.model_tier,
            "--distributed",
            "--split-scheme", "ring",
            "--use-fp8",
        ]
        
        return " ".join(cmd)
    
    def submit(self, job: VideoJob) -> Dict[str, Any]:
        """
        Submit a video generation job.
        
        Returns:
            Dict with job_id, mode, node, and command
        """
        mode = self.determine_mode(job)
        
        if mode == ExecutionMode.FACTORY:
            node = self.select_node(mode)
            command = self.dispatch_factory(job, node)
            return {
                "job_id": job.job_id,
                "mode": mode.value,
                "node": node.name,
                "command": command,
            }
        else:
            command = self.dispatch_titan(job)
            return {
                "job_id": job.job_id,
                "mode": mode.value,
                "nodes": [n.name for n in self.nodes],
                "command": command,
            }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="The Auteur System Orchestrator")
    parser.add_argument("--prompt", required=True, help="Video prompt")
    parser.add_argument("--model-tier", default="workhorse", 
                        choices=["validator", "workhorse", "heavy"])
    parser.add_argument("--job-id", default=None, help="Job ID (auto-generated if not provided)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    
    args = parser.parse_args()
    
    # Generate job ID
    import uuid
    job_id = args.job_id or str(uuid.uuid4())[:8]
    
    # Create job
    job = VideoJob(
        job_id=job_id,
        prompt=args.prompt,
        model_tier=args.model_tier,
    )
    
    # Submit
    orchestrator = Orchestrator()
    result = orchestrator.submit(job)
    
    print(f"\n{'='*60}")
    print(f"Job ID: {result['job_id']}")
    print(f"Mode: {result['mode']}")
    print(f"Command: {result['command']}")
    print(f"{'='*60}\n")
    
    if not args.dry_run:
        print("[Orchestrator] Executing job...")
        # subprocess.run(result['command'], shell=True)


if __name__ == "__main__":
    main()

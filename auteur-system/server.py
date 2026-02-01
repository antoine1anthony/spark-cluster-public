#!/usr/bin/env python3
"""
The Auteur System - Async Job Service (Hardened v2)
Runs on Spark4 (Brawn Node) - exposes video generation as a REST API.

Architecture:
- Single-worker async job queue (1 GPU = 1 Job) - ENFORCED
- Bearer token authentication
- Supports both T2V (JSON) and I2V (Multipart with image)
- Jobs stored in /output with job_id subdirectories
- Full metadata tracking with job.json per job
- TTL-based cleanup for completed/failed jobs
- Idempotency via request_id

Endpoints:
- POST /jobs          - Submit job (JSON or Multipart)
- POST /jobs/json     - Submit T2V job (JSON only)
- GET  /jobs          - List jobs
- GET  /jobs/{id}     - Get job status
- GET  /jobs/{id}/download - Stream result video
- DELETE /jobs/{id}   - Cleanup job artifacts (idempotent)
- GET  /health        - Liveness check with disk/queue stats
"""

import os
import sys
import uuid
import asyncio
import shutil
import logging
import time
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# =============================================================================
# Configuration
# =============================================================================

# Version info
WORKER_VERSION = "2.1.0-hardened"

# Auth token - set via environment variable
AUTH_TOKEN = os.environ.get("AUTEUR_TOKEN", "change-me-in-production")

# Output directory - where jobs and videos are stored
OUTPUT_DIR = Path(os.environ.get("AUTEUR_OUTPUT_DIR", "/output"))

# Optimization defaults - EXPLICITLY set
DEFAULT_NO_OFFLOAD = True  # GB10 has 128GB unified memory - no need to offload
DEFAULT_COMPILE = False    # Compile adds latency for first job, enable consciously

# TTL for job cleanup (24 hours)
JOB_TTL_HOURS = int(os.environ.get("AUTEUR_JOB_TTL_HOURS", "24"))

# Cleanup interval (15 minutes)
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("AUTEUR_CLEANUP_INTERVAL", "900"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[AuteurWorker] %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("auteur.server")

# =============================================================================
# Startup Checks
# =============================================================================

def check_ffprobe() -> bool:
    """Check if ffprobe is available - FAIL FAST if missing."""
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            logger.info(f"✓ ffprobe available: {version}")
            return True
        else:
            logger.error("✗ ffprobe not working properly")
            return False
    except FileNotFoundError:
        logger.error("✗ ffprobe not found - CRITICAL: video validation will fail")
        return False
    except Exception as e:
        logger.error(f"✗ ffprobe check failed: {e}")
        return False

def run_ffprobe(video_path: str) -> Optional[Dict[str, Any]]:
    """Run ffprobe on a video and return parsed metadata."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,bit_rate",
            "-show_entries", "format=duration,size",
            "-of", "json",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.warning(f"ffprobe failed: {result.stderr}")
            return None
        
        data = json.loads(result.stdout)
        
        # Parse into flat structure
        summary = {
            "valid": True,
            "path": video_path,
        }
        
        if data.get("streams"):
            stream = data["streams"][0]
            summary["codec"] = stream.get("codec_name")
            summary["width"] = int(stream.get("width", 0))
            summary["height"] = int(stream.get("height", 0))
            summary["frame_rate"] = stream.get("avg_frame_rate")
            summary["nb_frames"] = int(stream.get("nb_frames", 0)) if stream.get("nb_frames") else None
            summary["bit_rate"] = int(stream.get("bit_rate", 0)) if stream.get("bit_rate") else None
        
        if data.get("format"):
            fmt = data["format"]
            summary["duration"] = float(fmt.get("duration", 0)) if fmt.get("duration") else None
            summary["size_bytes"] = int(fmt.get("size", 0)) if fmt.get("size") else None
        
        return summary
        
    except Exception as e:
        logger.warning(f"ffprobe error: {e}")
        return None

# =============================================================================
# Models
# =============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelTier(str, Enum):
    VALIDATOR = "validator"      # SVD - Image-to-Video
    WORKHORSE = "workhorse"      # HunyuanVideo - Text-to-Video
    EXPERIMENTAL = "experimental" # CogVideoX - Text-to-Video (unstable)

@dataclass
class JobMetadata:
    """Full job metadata saved to job.json."""
    job_id: str
    request_id: Optional[str]
    status: str
    model_tier: str
    prompt: Optional[str]
    prompt_hash: Optional[str]  # SHA256 of prompt for privacy
    input_image_path: Optional[str]
    num_frames: int
    num_steps: int
    no_offload_effective: bool
    compile_effective: bool
    effective_backend: Optional[str]  # flash/xformers/sdpa
    
    # Timings
    enqueue_time: str
    start_time: Optional[str]
    end_time: Optional[str]
    
    # Results
    output_video_path: Optional[str]
    error: Optional[str]
    ffprobe_summary: Optional[Dict[str, Any]]
    
    # System
    peak_rss_gb: Optional[float]
    worker_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Job:
    id: str
    status: JobStatus
    model_tier: ModelTier
    request_id: Optional[str] = None
    prompt: Optional[str] = None
    input_image_path: Optional[str] = None
    num_frames: Optional[int] = None
    num_steps: Optional[int] = None
    no_offload: bool = DEFAULT_NO_OFFLOAD
    compile_model: bool = DEFAULT_COMPILE
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_video_path: Optional[str] = None
    error: Optional[str] = None
    progress: int = 0
    effective_backend: Optional[str] = None
    peak_rss_gb: Optional[float] = None
    ffprobe_summary: Optional[Dict[str, Any]] = None
    cancel_requested: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["model_tier"] = self.model_tier.value
        return d

class JobRequest(BaseModel):
    model_tier: ModelTier = ModelTier.WORKHORSE
    prompt: Optional[str] = None
    num_frames: Optional[int] = None
    num_steps: Optional[int] = None
    no_offload: Optional[bool] = None
    compile_model: Optional[bool] = None
    request_id: Optional[str] = None  # Client-provided idempotency key

# =============================================================================
# Application State
# =============================================================================

app = FastAPI(
    title="Auteur Video Generation Service",
    description="Async video generation API for DGX Spark GB10 (Hardened v2)",
    version=WORKER_VERSION
)

security = HTTPBearer()

# In-memory job store (jobs persist on disk too)
jobs: Dict[str, Job] = {}

# Request ID to Job ID mapping for idempotency
request_id_map: Dict[str, str] = {}

# Job queue
job_queue: asyncio.Queue = asyncio.Queue()

# Worker state
worker_running = False
startup_time = datetime.utcnow()
last_error: Optional[str] = None
running_job_id: Optional[str] = None
ffprobe_available = False

# Pipeline cache (lazy loaded)
_pipeline_cache: Dict[str, Any] = {}

# =============================================================================
# Authentication
# =============================================================================

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify Bearer token."""
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# =============================================================================
# Disk Utilities
# =============================================================================

def get_disk_usage() -> Dict[str, float]:
    """Get disk usage stats for OUTPUT_DIR."""
    try:
        stat = os.statvfs(OUTPUT_DIR)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        used_gb = total_gb - free_gb
        return {
            "total_gb": round(total_gb, 2),
            "free_gb": round(free_gb, 2),
            "used_gb": round(used_gb, 2),
        }
    except Exception as e:
        logger.warning(f"Could not get disk usage: {e}")
        return {"total_gb": 0, "free_gb": 0, "used_gb": 0}

def get_hf_cache_size_gb() -> float:
    """Get Hugging Face cache size in GB."""
    hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    try:
        if not hf_home.exists():
            return 0.0
        total_size = sum(f.stat().st_size for f in hf_home.rglob("*") if f.is_file())
        return round(total_size / (1024**3), 2)
    except Exception as e:
        logger.warning(f"Could not get HF cache size: {e}")
        return -1.0  # -1 indicates error

def read_peak_rss_gb() -> Optional[float]:
    """Read peak RSS from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 2)
    except Exception:
        pass
    return None

# =============================================================================
# Job Metadata Management
# =============================================================================

def get_prompt_hash(prompt: Optional[str]) -> Optional[str]:
    """Get SHA256 hash of prompt for privacy-safe logging."""
    if not prompt:
        return None
    import hashlib
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]

def save_job_metadata(job: Job, phase: str = "update"):
    """Save job metadata to job.json."""
    job_dir = OUTPUT_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = JobMetadata(
        job_id=job.id,
        request_id=job.request_id,
        status=job.status.value,
        model_tier=job.model_tier.value,
        prompt=None,  # Don't store full prompt
        prompt_hash=get_prompt_hash(job.prompt),
        input_image_path=job.input_image_path,
        num_frames=job.num_frames or 0,
        num_steps=job.num_steps or 0,
        no_offload_effective=job.no_offload,
        compile_effective=job.compile_model,
        effective_backend=job.effective_backend,
        enqueue_time=job.created_at,
        start_time=job.started_at,
        end_time=job.completed_at,
        output_video_path=job.output_video_path,
        error=job.error,
        ffprobe_summary=job.ffprobe_summary,
        peak_rss_gb=job.peak_rss_gb,
        worker_version=WORKER_VERSION,
    )
    
    metadata_path = job_dir / "job.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2, default=str)
    
    logger.debug(f"Saved job metadata ({phase}): {metadata_path}")

# =============================================================================
# Pipeline Management
# =============================================================================

def get_pipeline(model_tier: ModelTier, no_offload: bool, compile_model: bool):
    """
    Get or load pipeline for the specified tier.
    Caches pipeline to avoid reloading between jobs of the same tier.
    Returns (pipe, config, effective_backend).
    """
    global _pipeline_cache
    
    cache_key = f"{model_tier.value}_{no_offload}_{compile_model}"
    
    if cache_key in _pipeline_cache:
        logger.info(f"Using cached pipeline for {model_tier.value}")
        return _pipeline_cache[cache_key]
    
    # Clear previous cache to free memory
    if _pipeline_cache:
        logger.info("Clearing previous pipeline from cache...")
        _pipeline_cache.clear()
        import torch
        torch.cuda.empty_cache()
    
    # Import and load
    logger.info(f"Loading pipeline: {model_tier.value} (no_offload={no_offload}, compile={compile_model})")
    
    # Import generate module
    sys.path.insert(0, str(Path(__file__).parent))
    from generate import load_model, _log_attention_backend
    
    pipe, config = load_model(
        model_tier.value,
        use_fp8=True,
        no_offload=no_offload,
        compile_model=compile_model
    )
    
    # Detect effective backend
    effective_backend = "unknown"
    if hasattr(pipe, 'transformer'):
        attn_procs = getattr(pipe.transformer, 'attn_processors', {})
        if attn_procs:
            proc_types = set(type(p).__name__ for p in attn_procs.values())
            if any('Flash' in t for t in proc_types):
                effective_backend = "flash_attn"
            elif any('XFormers' in t for t in proc_types):
                effective_backend = "xformers"
            else:
                effective_backend = "sdpa"
    
    _pipeline_cache[cache_key] = (pipe, config, effective_backend)
    logger.info(f"Pipeline loaded and cached: {model_tier.value} (backend: {effective_backend})")
    
    return pipe, config, effective_backend

# =============================================================================
# Job Processing
# =============================================================================

async def process_job(job: Job):
    """Process a single job - runs in the worker loop."""
    global running_job_id, last_error
    
    running_job_id = job.id
    logger.info(f"Processing job {job.id} ({job.model_tier.value})")
    
    # Check for early cancellation (cancelled while pending in queue)
    if job.cancel_requested:
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow().isoformat()
        job.error = "Cancelled before execution started"
        save_job_metadata(job, "cancelled")
        logger.info(f"Job {job.id} was cancelled before processing")
        running_job_id = None
        return
    
    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow().isoformat()
    save_job_metadata(job, "started")
    
    # Create job output directory
    job_dir = OUTPUT_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get or load pipeline
        pipe, config, effective_backend = get_pipeline(
            job.model_tier,
            job.no_offload,
            job.compile_model
        )
        job.effective_backend = effective_backend
        
        # Check for cancellation after model loading (can take time)
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow().isoformat()
            job.error = "Cancelled after model loading"
            save_job_metadata(job, "cancelled")
            logger.info(f"Job {job.id} cancelled after model loading")
            running_job_id = None
            return
        
        # Import generate function
        sys.path.insert(0, str(Path(__file__).parent))
        from generate import generate_video
        
        # Determine frames and steps with explicit defaults
        num_frames = job.num_frames or config.get("default_frames", 25)
        num_steps = job.num_steps or config.get("default_steps", 25)
        job.num_frames = num_frames
        job.num_steps = num_steps
        
        # Generate video
        # Note: For full cancellation support during generation, the generate_video
        # function would need to accept a callback/flag. Current implementation
        # only checks before/after generation.
        result = generate_video(
            pipe=pipe,
            config=config,
            prompt=job.prompt or "",
            output_dir=str(job_dir),
            num_frames=num_frames,
            num_inference_steps=num_steps,
            input_image=job.input_image_path,
            monitor=None,
        )
        
        # Check for post-generation cancellation (user doesn't want result)
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow().isoformat()
            job.error = "Cancelled after generation completed"
            save_job_metadata(job, "cancelled")
            logger.info(f"Job {job.id} cancelled (generation completed but result discarded)")
            running_job_id = None
            return
        
        # Capture peak RSS
        job.peak_rss_gb = read_peak_rss_gb()
        
        if result["success"]:
            job.status = JobStatus.COMPLETED
            job.output_video_path = result.get("video_path")
            job.progress = 100
            
            # Run ffprobe validation
            if job.output_video_path and ffprobe_available:
                job.ffprobe_summary = run_ffprobe(job.output_video_path)
                if job.ffprobe_summary:
                    logger.info(f"Job {job.id} video: {job.ffprobe_summary.get('width')}x{job.ffprobe_summary.get('height')}, "
                               f"{job.ffprobe_summary.get('nb_frames')} frames, "
                               f"{job.ffprobe_summary.get('size_bytes', 0) / 1024:.1f} KB")
            
            logger.info(f"Job {job.id} completed: {job.output_video_path}")
        else:
            job.status = JobStatus.FAILED
            job.error = result.get("error", "Unknown error")
            last_error = f"Job {job.id}: {job.error}"
            logger.error(f"Job {job.id} failed: {job.error}")
            
    except Exception as e:
        import traceback
        job.status = JobStatus.FAILED
        job.error = str(e)
        last_error = f"Job {job.id}: {e}"
        logger.error(f"Job {job.id} failed with exception: {e}")
        traceback.print_exc()
    
    job.completed_at = datetime.utcnow().isoformat()
    save_job_metadata(job, "completed")
    running_job_id = None

async def worker_loop():
    """Single worker loop - processes jobs one at a time."""
    global worker_running
    worker_running = True
    logger.info("Worker loop started - worker_count=1 (ENFORCED)")
    
    while True:
        try:
            # Wait for a job
            job_id = await job_queue.get()
            
            if job_id not in jobs:
                logger.warning(f"Job {job_id} not found in store, skipping")
                continue
            
            job = jobs[job_id]
            
            # Process the job (blocking - single worker)
            await process_job(job)
            
            job_queue.task_done()
            
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            await asyncio.sleep(1)

async def cleanup_loop():
    """Background task to clean up old completed/failed jobs."""
    logger.info(f"Cleanup loop started - TTL: {JOB_TTL_HOURS}h, interval: {CLEANUP_INTERVAL_SECONDS}s")
    
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            
            now = datetime.utcnow()
            ttl_threshold = now - timedelta(hours=JOB_TTL_HOURS)
            
            jobs_to_delete = []
            
            for job_id, job in list(jobs.items()):
                # Only clean up completed or failed jobs
                if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    continue
                
                # Must have completed_at timestamp
                if not job.completed_at:
                    continue
                
                try:
                    completed_time = datetime.fromisoformat(job.completed_at.replace('Z', '+00:00').replace('+00:00', ''))
                    if completed_time < ttl_threshold:
                        jobs_to_delete.append(job_id)
                except Exception as e:
                    logger.warning(f"Could not parse completed_at for job {job_id}: {e}")
                    continue
            
            for job_id in jobs_to_delete:
                try:
                    job_dir = OUTPUT_DIR / job_id
                    if job_dir.exists():
                        shutil.rmtree(job_dir)
                    
                    # Remove from request_id map
                    job = jobs.get(job_id)
                    if job and job.request_id and job.request_id in request_id_map:
                        del request_id_map[job.request_id]
                    
                    del jobs[job_id]
                    logger.info(f"TTL cleanup: deleted job {job_id}")
                except Exception as e:
                    logger.warning(f"TTL cleanup failed for job {job_id}: {e}")
            
            if jobs_to_delete:
                logger.info(f"TTL cleanup: removed {len(jobs_to_delete)} jobs")
                
        except Exception as e:
            logger.error(f"Cleanup loop error: {e}")

# =============================================================================
# API Endpoints
# =============================================================================

@app.on_event("startup")
async def startup():
    """Start the worker loop and cleanup loop on startup."""
    global ffprobe_available
    
    # Check ffprobe - FAIL FAST
    ffprobe_available = check_ffprobe()
    if not ffprobe_available:
        logger.error("CRITICAL: ffprobe not available - video validation disabled")
    
    # Start worker loop
    asyncio.create_task(worker_loop())
    
    # Start cleanup loop
    asyncio.create_task(cleanup_loop())
    
    logger.info(f"Auteur Worker Service v{WORKER_VERSION} started")
    logger.info(f"worker_count=1 (ENFORCED)")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Default no_offload: {DEFAULT_NO_OFFLOAD}")
    logger.info(f"Default compile: {DEFAULT_COMPILE}")

@app.get("/health")
async def health():
    """Health check endpoint with comprehensive stats."""
    import torch
    
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    
    disk = get_disk_usage()
    hf_cache_gb = get_hf_cache_size_gb()
    uptime = (datetime.utcnow() - startup_time).total_seconds()
    
    # Disk health indicator: green (>20%), yellow (10-20%), red (<10%)
    disk_pct_free = (disk["free_gb"] / disk["total_gb"] * 100) if disk["total_gb"] > 0 else 0
    if disk_pct_free > 20:
        disk_health = "green"
    elif disk_pct_free > 10:
        disk_health = "yellow"
    else:
        disk_health = "red"
    
    return {
        "ok": True,
        "status": "healthy",
        "worker_running": worker_running,
        "worker_count": 1,  # Enforced
        "queue_depth": job_queue.qsize(),
        "running_job_id": running_job_id,
        "total_jobs": len(jobs),
        "disk_free_gb": disk["free_gb"],
        "disk_used_gb": disk["used_gb"],
        "disk_total_gb": disk["total_gb"],
        "disk_health": disk_health,
        "hf_cache_gb": hf_cache_gb,
        "uptime_s": int(uptime),
        "last_error": last_error,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "ffprobe_available": ffprobe_available,
        "default_no_offload": DEFAULT_NO_OFFLOAD,
        "default_compile": DEFAULT_COMPILE,
        "worker_version": WORKER_VERSION,
    }

@app.post("/jobs", dependencies=[Depends(verify_token)])
async def create_job(
    model_tier: ModelTier = Form(ModelTier.WORKHORSE),
    prompt: Optional[str] = Form(None),
    num_frames: Optional[int] = Form(None),
    num_steps: Optional[int] = Form(None),
    no_offload: Optional[bool] = Form(None),
    compile_model: Optional[bool] = Form(None),
    request_id: Optional[str] = Form(None),
    input_image: Optional[UploadFile] = File(None),
):
    """
    Create a new video generation job (multipart form).
    
    For T2V models (workhorse, experimental): provide prompt
    For I2V models (validator): provide input_image
    
    Include request_id for idempotency - if same request_id is sent again,
    returns the existing job instead of creating a new one.
    """
    # Idempotency check
    if request_id and request_id in request_id_map:
        existing_job_id = request_id_map[request_id]
        if existing_job_id in jobs:
            job = jobs[existing_job_id]
            logger.info(f"Idempotent request: returning existing job {existing_job_id} for request_id {request_id}")
            return {
                "job_id": existing_job_id,
                "status": job.status.value,
                "queue_position": 0 if job.status != JobStatus.PENDING else job_queue.qsize(),
                "idempotent": True,
            }
    
    job_id = str(uuid.uuid4())
    
    # Validate inputs
    if model_tier == ModelTier.VALIDATOR and not input_image:
        raise HTTPException(
            status_code=400,
            detail="validator tier (SVD) requires input_image"
        )
    
    if model_tier in [ModelTier.WORKHORSE, ModelTier.EXPERIMENTAL] and not prompt:
        raise HTTPException(
            status_code=400,
            detail=f"{model_tier.value} tier requires a text prompt"
        )
    
    # Create job directory
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save input image if provided
    input_image_path = None
    if input_image:
        input_image_path = str(job_dir / "input_image.jpg")
        with open(input_image_path, "wb") as f:
            content = await input_image.read()
            f.write(content)
        logger.info(f"Saved input image for job {job_id}: {input_image_path}")
    
    # Create job with explicit defaults
    job = Job(
        id=job_id,
        status=JobStatus.PENDING,
        model_tier=model_tier,
        request_id=request_id,
        prompt=prompt,
        input_image_path=input_image_path,
        num_frames=num_frames,
        num_steps=num_steps,
        no_offload=no_offload if no_offload is not None else DEFAULT_NO_OFFLOAD,
        compile_model=compile_model if compile_model is not None else DEFAULT_COMPILE,
    )
    
    jobs[job_id] = job
    
    # Track request_id for idempotency
    if request_id:
        request_id_map[request_id] = job_id
    
    # Save initial metadata
    save_job_metadata(job, "created")
    
    # Add to queue
    await job_queue.put(job_id)
    
    logger.info(f"Job {job_id} created and queued ({model_tier.value})")
    
    return {
        "job_id": job_id,
        "status": job.status.value,
        "queue_position": job_queue.qsize(),
        "idempotent": False,
    }

@app.post("/jobs/json", dependencies=[Depends(verify_token)])
async def create_job_json(request: JobRequest):
    """
    Create a new T2V job using JSON body (no image upload).
    Convenience endpoint for text-to-video jobs.
    
    Include request_id for idempotency.
    """
    # Idempotency check
    if request.request_id and request.request_id in request_id_map:
        existing_job_id = request_id_map[request.request_id]
        if existing_job_id in jobs:
            job = jobs[existing_job_id]
            logger.info(f"Idempotent request: returning existing job {existing_job_id}")
            return {
                "job_id": existing_job_id,
                "status": job.status.value,
                "queue_position": 0 if job.status != JobStatus.PENDING else job_queue.qsize(),
                "idempotent": True,
            }
    
    if request.model_tier == ModelTier.VALIDATOR:
        raise HTTPException(
            status_code=400,
            detail="validator tier requires image upload - use multipart /jobs endpoint"
        )
    
    if not request.prompt:
        raise HTTPException(
            status_code=400,
            detail="prompt is required for T2V models"
        )
    
    job_id = str(uuid.uuid4())
    
    # Create job directory
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Create job with explicit defaults
    job = Job(
        id=job_id,
        status=JobStatus.PENDING,
        model_tier=request.model_tier,
        request_id=request.request_id,
        prompt=request.prompt,
        num_frames=request.num_frames,
        num_steps=request.num_steps,
        no_offload=request.no_offload if request.no_offload is not None else DEFAULT_NO_OFFLOAD,
        compile_model=request.compile_model if request.compile_model is not None else DEFAULT_COMPILE,
    )
    
    jobs[job_id] = job
    
    # Track request_id for idempotency
    if request.request_id:
        request_id_map[request.request_id] = job_id
    
    # Save initial metadata
    save_job_metadata(job, "created")
    
    # Add to queue
    await job_queue.put(job_id)
    
    logger.info(f"Job {job_id} created via JSON ({request.model_tier.value})")
    
    return {
        "job_id": job_id,
        "status": job.status.value,
        "queue_position": job_queue.qsize(),
        "idempotent": False,
    }

@app.get("/jobs/{job_id}", dependencies=[Depends(verify_token)])
async def get_job(job_id: str):
    """Get job status and details."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return job.to_dict()

@app.get("/jobs/{job_id}/download", dependencies=[Depends(verify_token)])
async def download_job_result(job_id: str):
    """Download the generated video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job.status.value})"
        )
    
    if not job.output_video_path or not Path(job.output_video_path).exists():
        raise HTTPException(status_code=404, detail="Output video not found")
    
    return FileResponse(
        job.output_video_path,
        media_type="video/mp4",
        filename=f"auteur_{job_id}.mp4"
    )

@app.delete("/jobs/{job_id}", dependencies=[Depends(verify_token)])
async def delete_job(job_id: str):
    """
    Delete job and cleanup artifacts.
    
    - Idempotent: returns 200 even if job already deleted
    - Safe: refuses to delete running jobs
    - Thorough: removes entire job directory
    """
    # Idempotent - return success if job doesn't exist
    if job_id not in jobs:
        return {"status": "deleted", "job_id": job_id, "message": "Job not found or already deleted"}
    
    job = jobs[job_id]
    
    # Don't delete running jobs - this is a safety check
    if job.status == JobStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete running job. Use cancel endpoint or wait for completion."
        )
    
    # Cleanup entire job directory (includes mp4, input image, job.json, any temps)
    job_dir = OUTPUT_DIR / job_id
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir)
            logger.info(f"Deleted job directory: {job_dir}")
        except Exception as e:
            logger.warning(f"Could not fully delete job directory {job_dir}: {e}")
    
    # Remove from request_id map
    if job.request_id and job.request_id in request_id_map:
        del request_id_map[job.request_id]
    
    # Remove from store
    del jobs[job_id]
    
    return {"status": "deleted", "job_id": job_id}

@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(verify_token)])
async def cancel_job(job_id: str):
    """
    Request cancellation of a job.
    
    - For PENDING jobs: immediately marks as CANCELLED
    - For RUNNING jobs: sets cancel_requested flag (checked between steps)
    - For COMPLETED/FAILED/CANCELLED jobs: no-op with appropriate message
    
    Note: Cancellation is cooperative - running jobs will stop at the next
    checkpoint. There's no way to forcefully abort mid-generation without
    risking GPU state corruption.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status == JobStatus.PENDING:
        # Pending jobs can be cancelled immediately
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow().isoformat()
        job.error = "Cancelled by user before execution"
        save_job_metadata(job, "cancelled")
        logger.info(f"Job {job_id} cancelled (was pending)")
        return {
            "status": "cancelled",
            "job_id": job_id,
            "message": "Job cancelled successfully (was pending)"
        }
    
    elif job.status == JobStatus.RUNNING:
        # Running jobs: set cancel flag, will be checked between steps
        job.cancel_requested = True
        logger.info(f"Cancel requested for running job {job_id}")
        return {
            "status": "cancel_requested",
            "job_id": job_id,
            "message": "Cancellation requested. Job will stop at next checkpoint."
        }
    
    elif job.status == JobStatus.CANCELLED:
        return {
            "status": "already_cancelled",
            "job_id": job_id,
            "message": "Job was already cancelled"
        }
    
    else:
        # COMPLETED or FAILED
        return {
            "status": "no_op",
            "job_id": job_id,
            "message": f"Job already finished with status: {job.status.value}"
        }

@app.get("/jobs", dependencies=[Depends(verify_token)])
async def list_jobs(status: Optional[JobStatus] = None, limit: int = 50):
    """List all jobs, optionally filtered by status."""
    result = []
    
    for job in jobs.values():
        if status is None or job.status == status:
            result.append(job.to_dict())
    
    # Sort by created_at descending
    result.sort(key=lambda x: x["created_at"], reverse=True)
    
    return result[:limit]

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auteur Video Generation Worker (Hardened)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers (MUST be 1)")
    
    args = parser.parse_args()
    
    # ENFORCE single worker
    if args.workers != 1:
        logger.error(f"CRITICAL: workers must be 1, got {args.workers}. Forcing workers=1.")
        args.workers = 1
    
    logger.info(f"Starting Auteur Worker v{WORKER_VERSION} on {args.host}:{args.port}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Default no_offload: {DEFAULT_NO_OFFLOAD}")
    logger.info(f"Default compile: {DEFAULT_COMPILE}")
    logger.info(f"worker_count=1 (ENFORCED)")
    
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        workers=1,  # ENFORCED
        log_level="info",
    )

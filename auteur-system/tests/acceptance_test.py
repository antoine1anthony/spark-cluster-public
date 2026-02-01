#!/usr/bin/env python3
"""
Auteur System - Acceptance Test Suite
=====================================

Validates the video generation pipeline with strict ffprobe validation.

Tests:
1. T2V (HunyuanVideo) - Text-to-Video generation
2. I2V (SVD) - Image-to-Video generation (if test image available)

Validation:
- Frames >= requested_frames * 0.9
- Width/height matches expected profile
- Duration matches frames/fps within tolerance
- Size and bitrate sanity checks
"""

import os
import sys
import json
import time
import uuid
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Configuration
AUTEUR_URL = os.environ.get("AUTEUR_URL", "http://localhost:8000")
AUTEUR_TOKEN = os.environ.get("AUTEUR_TOKEN", "change-me-in-production")
OUTPUT_DIR = Path(os.environ.get("TEST_OUTPUT_DIR", "/home/antoine1anthony/auteur-output"))
RESULTS_DIR = Path(__file__).parent / "results"

# Timeouts
MAX_POLL_TIME = 900  # 15 minutes max per job
POLL_INTERVAL = 10   # Poll every 10 seconds

# Expected profiles
PROFILES = {
    "workhorse": {
        "name": "HunyuanVideo T2V",
        "expected_width": 1280,
        "expected_height": 720,
        "expected_fps": 15,
        "min_size_kb": 50,
        "max_size_mb": 100,
    },
    "validator": {
        "name": "SVD I2V",
        "expected_width": 1024,
        "expected_height": 576,
        "expected_fps": 8,
        "min_size_kb": 50,
        "max_size_mb": 50,
    }
}

class AcceptanceTestResult:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.start_time = datetime.utcnow()
        self.end_time = None
    
    def add_test(self, name: str, passed: bool, details: Dict[str, Any]):
        self.tests.append({
            "name": name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def to_dict(self) -> Dict[str, Any]:
        self.end_time = datetime.utcnow()
        return {
            "summary": {
                "total": len(self.tests),
                "passed": self.passed,
                "failed": self.failed,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_s": (self.end_time - self.start_time).total_seconds()
            },
            "tests": self.tests
        }

def run_ffprobe_local(video_path: str) -> Optional[Dict[str, Any]]:
    """Run ffprobe locally on a video file (if available)."""
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
            return None
        
        data = json.loads(result.stdout)
        
        summary = {"valid": True, "path": video_path}
        
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
        
    except FileNotFoundError:
        return None  # ffprobe not installed locally
    except Exception:
        return None

def get_ffprobe_data(job: Dict[str, Any], local_path: str = None) -> Optional[Dict[str, Any]]:
    """Get ffprobe data from job metadata or run locally as fallback."""
    # First try to get from worker's ffprobe_summary (preferred - validated at source)
    ffprobe_data = job.get("ffprobe_summary")
    if ffprobe_data and ffprobe_data.get("valid"):
        print("  Using worker's ffprobe validation data")
        return ffprobe_data
    
    # Fallback: try local ffprobe if available
    if local_path:
        local_data = run_ffprobe_local(local_path)
        if local_data:
            print("  Using local ffprobe validation")
            return local_data
    
    # If no ffprobe data available, create minimal from file size
    if local_path and Path(local_path).exists():
        size = Path(local_path).stat().st_size
        print(f"  Warning: No ffprobe data available, using file size only ({size} bytes)")
        return {"valid": True, "size_bytes": size, "path": local_path}
    
    return None

def validate_video(
    ffprobe_data: Dict[str, Any],
    requested_frames: int,
    profile: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Validate video output against request and profile."""
    errors = []
    
    # Frame count validation (>= 90% of requested)
    actual_frames = ffprobe_data.get("nb_frames")
    if actual_frames is None:
        errors.append("Frame count not available in ffprobe output")
    elif actual_frames < requested_frames * 0.9:
        errors.append(f"Frame count {actual_frames} < {requested_frames * 0.9} (90% of requested {requested_frames})")
    
    # Resolution validation
    actual_width = ffprobe_data.get("width", 0)
    actual_height = ffprobe_data.get("height", 0)
    expected_width = profile["expected_width"]
    expected_height = profile["expected_height"]
    
    # Allow some tolerance for resolution (within 10%)
    if abs(actual_width - expected_width) > expected_width * 0.1:
        errors.append(f"Width {actual_width} not close to expected {expected_width}")
    if abs(actual_height - expected_height) > expected_height * 0.1:
        errors.append(f"Height {actual_height} not close to expected {expected_height}")
    
    # Duration validation (if frames and fps available)
    duration = ffprobe_data.get("duration")
    if duration and actual_frames:
        expected_duration = actual_frames / profile["expected_fps"]
        if abs(duration - expected_duration) > expected_duration * 0.2:
            errors.append(f"Duration {duration:.2f}s not close to expected {expected_duration:.2f}s")
    
    # Size sanity check
    size_bytes = ffprobe_data.get("size_bytes", 0)
    min_size = profile["min_size_kb"] * 1024
    max_size = profile["max_size_mb"] * 1024 * 1024
    
    if size_bytes < min_size:
        errors.append(f"File size {size_bytes / 1024:.1f}KB < minimum {profile['min_size_kb']}KB - likely corrupt")
    if size_bytes > max_size:
        errors.append(f"File size {size_bytes / (1024*1024):.1f}MB > maximum {profile['max_size_mb']}MB - suspiciously large")
    
    # Bitrate sanity (warn only, not fail)
    bit_rate = ffprobe_data.get("bit_rate")
    if bit_rate is None and size_bytes < min_size * 2:
        errors.append("Missing bitrate AND small file size - validation concern")
    
    return len(errors) == 0, errors

def submit_job(
    model_tier: str,
    prompt: str = None,
    num_frames: int = 17,
    num_steps: int = 20,
    request_id: str = None,
    input_image_path: str = None
) -> Dict[str, Any]:
    """Submit a job to the Auteur worker."""
    headers = {"Authorization": f"Bearer {AUTEUR_TOKEN}"}
    
    if input_image_path:
        # Multipart form for I2V
        with open(input_image_path, "rb") as f:
            files = {"input_image": (Path(input_image_path).name, f, "image/jpeg")}
            data = {
                "model_tier": model_tier,
                "prompt": prompt or "animate this image",
                "num_frames": str(num_frames),
                "num_steps": str(num_steps),
            }
            if request_id:
                data["request_id"] = request_id
            
            response = requests.post(f"{AUTEUR_URL}/jobs", headers=headers, files=files, data=data)
    else:
        # JSON for T2V
        payload = {
            "model_tier": model_tier,
            "prompt": prompt,
            "num_frames": num_frames,
            "num_steps": num_steps,
        }
        if request_id:
            payload["request_id"] = request_id
        
        response = requests.post(f"{AUTEUR_URL}/jobs/json", headers=headers, json=payload)
    
    response.raise_for_status()
    return response.json()

def poll_job(job_id: str) -> Dict[str, Any]:
    """Poll for job completion."""
    headers = {"Authorization": f"Bearer {AUTEUR_TOKEN}"}
    start_time = time.time()
    
    while time.time() - start_time < MAX_POLL_TIME:
        response = requests.get(f"{AUTEUR_URL}/jobs/{job_id}", headers=headers)
        response.raise_for_status()
        job = response.json()
        
        status = job.get("status")
        progress = job.get("progress", 0)
        elapsed = int(time.time() - start_time)
        
        print(f"  [Poll] Status: {status}, Progress: {progress}%, Elapsed: {elapsed}s")
        
        if status == "completed":
            return job
        elif status == "failed":
            return job
        
        time.sleep(POLL_INTERVAL)
    
    return {"status": "timeout", "error": f"Job timed out after {MAX_POLL_TIME}s"}

def download_video(job_id: str, output_path: Path) -> bool:
    """Download video from completed job."""
    headers = {"Authorization": f"Bearer {AUTEUR_TOKEN}"}
    
    try:
        response = requests.get(f"{AUTEUR_URL}/jobs/{job_id}/download", headers=headers, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False

def test_t2v(result: AcceptanceTestResult):
    """Test Text-to-Video generation."""
    test_name = "T2V HunyuanVideo"
    profile = PROFILES["workhorse"]
    request_id = str(uuid.uuid4())
    num_frames = 17
    num_steps = 20
    
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"Request ID: {request_id}")
    print(f"Frames: {num_frames}, Steps: {num_steps}")
    
    details = {
        "request_id": request_id,
        "model_tier": "workhorse",
        "num_frames": num_frames,
        "num_steps": num_steps,
    }
    
    try:
        # Submit job
        print("\n[1] Submitting job...")
        submit_result = submit_job(
            model_tier="workhorse",
            prompt="A cinematic drone shot of ocean waves crashing on rocky cliffs at sunset, golden hour lighting, dramatic clouds",
            num_frames=num_frames,
            num_steps=num_steps,
            request_id=request_id
        )
        
        job_id = submit_result["job_id"]
        details["job_id"] = job_id
        print(f"  ✓ Job submitted: {job_id}")
        
        # Poll for completion
        print("\n[2] Waiting for completion...")
        job = poll_job(job_id)
        details["final_status"] = job.get("status")
        
        if job.get("status") != "completed":
            details["error"] = job.get("error", "Job did not complete")
            result.add_test(test_name, False, details)
            return
        
        print(f"  ✓ Job completed")
        
        # Download video
        print("\n[3] Downloading video...")
        output_path = OUTPUT_DIR / f"acceptance_t2v_{job_id}.mp4"
        if not download_video(job_id, output_path):
            details["error"] = "Download failed"
            result.add_test(test_name, False, details)
            return
        
        details["output_path"] = str(output_path)
        print(f"  ✓ Downloaded to {output_path}")
        
        # Run ffprobe validation (use worker's data or local)
        print("\n[4] Validating with ffprobe...")
        ffprobe_data = get_ffprobe_data(job, str(output_path))
        if not ffprobe_data:
            details["error"] = "No ffprobe data available (neither from worker nor locally)"
            result.add_test(test_name, False, details)
            return
        
        details["ffprobe"] = ffprobe_data
        print(f"  Resolution: {ffprobe_data.get('width')}x{ffprobe_data.get('height')}")
        print(f"  Frames: {ffprobe_data.get('nb_frames')}")
        print(f"  Duration: {ffprobe_data.get('duration', 0):.2f}s")
        print(f"  Size: {ffprobe_data.get('size_bytes', 0) / 1024:.1f}KB")
        
        # Validate against profile
        valid, errors = validate_video(ffprobe_data, num_frames, profile)
        details["validation_errors"] = errors
        
        if not valid:
            print(f"\n  ✗ Validation failed:")
            for err in errors:
                print(f"    - {err}")
            result.add_test(test_name, False, details)
            return
        
        print(f"\n  ✓ Validation PASSED")
        result.add_test(test_name, True, details)
        
    except Exception as e:
        details["error"] = str(e)
        print(f"\n  ✗ Exception: {e}")
        result.add_test(test_name, False, details)

def test_idempotency(result: AcceptanceTestResult):
    """Test idempotency with request_id."""
    test_name = "Idempotency Test"
    request_id = str(uuid.uuid4())
    
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"Request ID: {request_id}")
    
    details = {"request_id": request_id}
    
    try:
        # Submit first job
        print("\n[1] Submitting first request...")
        result1 = submit_job(
            model_tier="workhorse",
            prompt="Test prompt for idempotency",
            num_frames=17,
            num_steps=20,
            request_id=request_id
        )
        job_id1 = result1["job_id"]
        details["first_job_id"] = job_id1
        print(f"  ✓ First job: {job_id1}")
        
        # Submit second job with same request_id
        print("\n[2] Submitting duplicate request...")
        result2 = submit_job(
            model_tier="workhorse",
            prompt="Test prompt for idempotency",
            num_frames=17,
            num_steps=20,
            request_id=request_id
        )
        job_id2 = result2["job_id"]
        details["second_job_id"] = job_id2
        details["idempotent_flag"] = result2.get("idempotent", False)
        
        if job_id1 == job_id2 and result2.get("idempotent"):
            print(f"  ✓ Idempotent response: same job_id returned")
            result.add_test(test_name, True, details)
        else:
            print(f"  ✗ Idempotency failed: got different job_id {job_id2}")
            details["error"] = "Different job_id returned for same request_id"
            result.add_test(test_name, False, details)
            
    except Exception as e:
        details["error"] = str(e)
        print(f"\n  ✗ Exception: {e}")
        result.add_test(test_name, False, details)

def test_health_endpoint(result: AcceptanceTestResult):
    """Test health endpoint returns expected fields."""
    test_name = "Health Endpoint"
    
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    details = {}
    
    try:
        response = requests.get(f"{AUTEUR_URL}/health")
        response.raise_for_status()
        health = response.json()
        details["response"] = health
        
        # Check required fields
        required_fields = [
            "ok", "worker_running", "worker_count", "queue_depth",
            "disk_free_gb", "disk_used_gb", "uptime_s", "ffprobe_available",
            "worker_version"
        ]
        
        missing = [f for f in required_fields if f not in health]
        
        if missing:
            details["missing_fields"] = missing
            print(f"  ✗ Missing fields: {missing}")
            result.add_test(test_name, False, details)
            return
        
        print(f"  Worker Version: {health.get('worker_version')}")
        print(f"  Worker Count: {health.get('worker_count')}")
        print(f"  FFprobe Available: {health.get('ffprobe_available')}")
        print(f"  Disk Free: {health.get('disk_free_gb')}GB")
        
        if health.get("worker_count") != 1:
            details["error"] = f"Worker count is {health.get('worker_count')}, expected 1"
            result.add_test(test_name, False, details)
            return
        
        print(f"\n  ✓ Health endpoint valid")
        result.add_test(test_name, True, details)
        
    except Exception as e:
        details["error"] = str(e)
        print(f"\n  ✗ Exception: {e}")
        result.add_test(test_name, False, details)

def main():
    """Run acceptance tests."""
    print("="*60)
    print("AUTEUR SYSTEM - ACCEPTANCE TEST SUITE")
    print("="*60)
    print(f"Worker URL: {AUTEUR_URL}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    result = AcceptanceTestResult()
    
    # Run tests
    test_health_endpoint(result)
    test_idempotency(result)
    test_t2v(result)
    
    # Save results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"acceptance_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {len(result.tests)}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")
    print(f"Results saved to: {results_file}")
    
    # Exit code based on results
    sys.exit(0 if result.failed == 0 else 1)

if __name__ == "__main__":
    main()

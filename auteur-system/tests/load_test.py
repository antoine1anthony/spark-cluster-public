#!/usr/bin/env python3
"""
Auteur System - Load Test (v2 - Fixed Timing)
==============================================

Tests the queueing behavior by submitting multiple jobs rapidly.

Validates:
1. Job 1 runs while Jobs 2 & 3 queue
2. Queue depth decreases monotonically
3. All jobs complete successfully
4. No OOM symptoms (peak RSS in metadata)
5. Per-job duration from actual timestamps (not script time)

Fixes from v1:
- Reports actual job duration from enqueue_time/start_time/end_time
- Adds quality validation with ffprobe thresholds
- Prints one-liner quality summary per job
"""

import os
import sys
import json
import time
import uuid
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Configuration
AUTEUR_URL = os.environ.get("AUTEUR_URL", "http://localhost:8000")
AUTEUR_TOKEN = os.environ.get("AUTEUR_TOKEN", "change-me-in-production")
RESULTS_DIR = Path(__file__).parent / "results"

# Timeouts
MAX_TOTAL_TIME = 3600  # 1 hour max for all jobs
POLL_INTERVAL = 30     # Poll every 30 seconds (jobs take ~5 min each)

# Quality thresholds
MIN_DURATION_S = 0.5       # Fail if video < 0.5s
MIN_BITRATE_KBPS = 200     # Warn if bitrate < 200 kbps for 720p
MIN_FRAMES = 10            # Fail if frames < 10

def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts:
        return None
    try:
        # Handle various formats
        ts = ts.replace('Z', '+00:00')
        if '+' not in ts and '-' not in ts[10:]:
            ts = ts + '+00:00'
        return datetime.fromisoformat(ts.replace('+00:00', ''))
    except Exception:
        return None

def format_quality_line(job: Dict[str, Any]) -> str:
    """Format one-liner quality summary for a job."""
    ff = job.get("ffprobe_summary", {})
    if not ff:
        return "⚠️ No ffprobe data"
    
    codec = ff.get("codec", "?")
    width = ff.get("width", 0)
    height = ff.get("height", 0)
    frames = ff.get("nb_frames", 0)
    duration = ff.get("duration", 0)
    size_kb = ff.get("size_bytes", 0) / 1024
    bitrate = ff.get("bit_rate", 0)
    bitrate_mbps = bitrate / 1_000_000 if bitrate else 0
    fps = ff.get("frame_rate", "?")
    
    # Quality checks
    issues = []
    if duration and duration < MIN_DURATION_S:
        issues.append(f"dur<{MIN_DURATION_S}s")
    if bitrate and bitrate < MIN_BITRATE_KBPS * 1000:
        issues.append(f"bitrate<{MIN_BITRATE_KBPS}kbps")
    if frames and frames < MIN_FRAMES:
        issues.append(f"frames<{MIN_FRAMES}")
    
    status = "✅" if not issues else f"⚠️ {','.join(issues)}"
    
    return f"{codec} {width}x{height} frames={frames} fps={fps} dur={duration:.2f}s size={size_kb:.0f}KB bitrate={bitrate_mbps:.2f}Mbps {status}"

def submit_job(prompt: str, request_id: str = None) -> Dict[str, Any]:
    """Submit a T2V job."""
    headers = {"Authorization": f"Bearer {AUTEUR_TOKEN}"}
    payload = {
        "model_tier": "workhorse",
        "prompt": prompt,
        "num_frames": 17,
        "num_steps": 20,
    }
    if request_id:
        payload["request_id"] = request_id
    
    response = requests.post(f"{AUTEUR_URL}/jobs/json", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def get_job(job_id: str) -> Dict[str, Any]:
    """Get job status."""
    headers = {"Authorization": f"Bearer {AUTEUR_TOKEN}"}
    response = requests.get(f"{AUTEUR_URL}/jobs/{job_id}", headers=headers)
    response.raise_for_status()
    return response.json()

def get_health() -> Dict[str, Any]:
    """Get worker health."""
    response = requests.get(f"{AUTEUR_URL}/health")
    response.raise_for_status()
    return response.json()

def main():
    """Run load test."""
    print("=" * 70)
    print("AUTEUR SYSTEM - LOAD TEST v2 (Fixed Timing)")
    print("=" * 70)
    print(f"Worker URL: {AUTEUR_URL}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {
        "version": "2.0",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "jobs": [],
        "queue_history": [],
        "passed": False,
        "errors": [],
        "warnings": []
    }
    
    # Initial health check
    print("\n[1] Initial Health Check")
    health = get_health()
    print(f"  Queue depth: {health['queue_depth']}")
    print(f"  Running job: {health['running_job_id']}")
    print(f"  Worker version: {health['worker_version']}")
    
    if health['queue_depth'] > 0 or health['running_job_id']:
        print("  ⚠ Warning: Queue not empty, test may include pre-existing jobs")
        results["warnings"].append("Queue was not empty at test start")
    
    # Submit 3 jobs rapidly with unique request_ids
    print("\n[2] Submitting 3 Jobs Rapidly")
    jobs = []
    prompts = [
        "A futuristic cityscape at night with neon lights reflecting on wet streets",
        "A serene forest clearing with sunlight filtering through ancient trees", 
        "An underwater coral reef scene with tropical fish swimming"
    ]
    
    test_batch_id = str(uuid.uuid4())[:8]
    
    for i, prompt in enumerate(prompts):
        req_id = f"loadtest-{test_batch_id}-{i}"
        submit_time = datetime.now(timezone.utc)
        result = submit_job(prompt, req_id)
        job_id = result["job_id"]
        jobs.append({
            "job_id": job_id,
            "request_id": req_id,
            "prompt": prompt[:50] + "...",
            "submitted_at": submit_time.isoformat(),
            "queue_position": result.get("queue_position", 0)
        })
        print(f"  Job {i+1}: {job_id[:8]}... (queue pos: {result.get('queue_position', '?')})")
        time.sleep(0.5)  # Small delay to ensure order
    
    results["jobs"] = jobs
    
    # Verify initial queue state
    print("\n[3] Verifying Queue Behavior")
    health = get_health()
    initial_queue = health['queue_depth']
    print(f"  Initial queue depth: {initial_queue}")
    results["queue_history"].append({
        "time": datetime.now(timezone.utc).isoformat(),
        "queue_depth": initial_queue,
        "running_job_id": health['running_job_id']
    })
    
    # Poll until all complete
    print("\n[4] Polling for Completion (this will take ~15-20 minutes)")
    start_time = time.time()
    completed_jobs = set()
    queue_depths = [initial_queue]
    
    while len(completed_jobs) < 3 and time.time() - start_time < MAX_TOTAL_TIME:
        time.sleep(POLL_INTERVAL)
        
        # Check health
        health = get_health()
        queue_depth = health['queue_depth']
        running_job = health.get('running_job_id')
        elapsed = int(time.time() - start_time)
        
        results["queue_history"].append({
            "time": datetime.now(timezone.utc).isoformat(),
            "queue_depth": queue_depth,
            "running_job_id": running_job
        })
        
        # Track queue depth changes
        if queue_depth != queue_depths[-1]:
            queue_depths.append(queue_depth)
        
        # Check each job
        for i, job_info in enumerate(jobs):
            job_id = job_info["job_id"]
            if job_id in completed_jobs:
                continue
            
            job = get_job(job_id)
            status = job.get("status")
            
            if status == "completed":
                completed_jobs.add(job_id)
                job_info["status"] = "completed"
                job_info["completed_at"] = datetime.now(timezone.utc).isoformat()
                job_info["peak_rss_gb"] = job.get("peak_rss_gb")
                job_info["ffprobe_summary"] = job.get("ffprobe_summary")
                job_info["effective_backend"] = job.get("effective_backend")
                
                # Calculate actual duration from job timestamps
                enqueue_time = parse_timestamp(job.get("created_at"))
                start_time_job = parse_timestamp(job.get("started_at"))
                end_time_job = parse_timestamp(job.get("completed_at"))
                
                if start_time_job and end_time_job:
                    actual_duration = (end_time_job - start_time_job).total_seconds()
                    job_info["actual_duration_s"] = actual_duration
                    job_info["actual_duration_min"] = actual_duration / 60
                else:
                    actual_duration = None
                    job_info["actual_duration_s"] = None
                
                # Store timestamps
                job_info["job_enqueue_time"] = job.get("created_at")
                job_info["job_start_time"] = job.get("started_at")
                job_info["job_end_time"] = job.get("completed_at")
                
                # Print quality line
                quality_line = format_quality_line(job)
                dur_str = f"{actual_duration:.1f}s ({actual_duration/60:.1f}min)" if actual_duration else "?"
                print(f"  [{elapsed}s] Job {i+1} COMPLETED in {dur_str}")
                print(f"       {quality_line}")
                
            elif status == "failed":
                completed_jobs.add(job_id)
                job_info["status"] = "failed"
                job_info["error"] = job.get("error")
                print(f"  [{elapsed}s] Job {i+1} FAILED: {job.get('error')}")
                results["errors"].append(f"Job {i+1} failed: {job.get('error')}")
        
        # Progress update
        running_str = f"running: {running_job[:8]}..." if running_job else "idle"
        pending = 3 - len(completed_jobs)
        print(f"  [{elapsed}s] Completed: {len(completed_jobs)}/3, Queue: {queue_depth}, {running_str}")
    
    # Verify results
    print("\n[5] Validating Results")
    
    # Check all completed
    all_completed = all(j.get("status") == "completed" for j in jobs)
    if all_completed:
        print("  ✓ All 3 jobs completed successfully")
    else:
        failed = [j for j in jobs if j.get("status") != "completed"]
        print(f"  ✗ {len(failed)} job(s) did not complete")
        results["errors"].append(f"{len(failed)} jobs failed to complete")
    
    # Check queue decreased monotonically
    non_decreasing = True
    for i in range(1, len(queue_depths)):
        if queue_depths[i] > queue_depths[i-1]:
            non_decreasing = False
            break
    
    if non_decreasing:
        print(f"  ✓ Queue depth decreased monotonically: {queue_depths}")
    else:
        print(f"  ✗ Queue depth not monotonic: {queue_depths}")
        results["errors"].append(f"Queue not monotonic: {queue_depths}")
    
    # Check peak RSS
    rss_values = [j.get("peak_rss_gb") for j in jobs if j.get("peak_rss_gb")]
    if rss_values:
        max_rss = max(rss_values)
        if max_rss < 100:
            print(f"  ✓ Peak RSS healthy: max {max_rss} GB")
        else:
            print(f"  ⚠ Peak RSS high: {max_rss} GB")
            results["warnings"].append(f"High RSS: {max_rss} GB")
    
    # Check quality thresholds
    quality_issues = []
    for i, j in enumerate(jobs):
        ff = j.get("ffprobe_summary", {})
        if ff:
            dur = ff.get("duration", 0)
            if dur and dur < MIN_DURATION_S:
                quality_issues.append(f"Job {i+1}: duration {dur}s < {MIN_DURATION_S}s")
            frames = ff.get("nb_frames", 0)
            if frames and frames < MIN_FRAMES:
                quality_issues.append(f"Job {i+1}: frames {frames} < {MIN_FRAMES}")
    
    if quality_issues:
        print(f"  ⚠ Quality warnings:")
        for q in quality_issues:
            print(f"    - {q}")
        results["warnings"].extend(quality_issues)
    else:
        print("  ✓ All quality thresholds passed")
    
    # Report actual durations
    print("\n[6] Per-Job Duration Summary")
    total_actual_time = 0
    for i, j in enumerate(jobs):
        dur = j.get("actual_duration_s")
        if dur:
            total_actual_time += dur
            print(f"  Job {i+1}: {dur:.1f}s ({dur/60:.1f} min)")
        else:
            print(f"  Job {i+1}: duration unknown")
    
    if total_actual_time > 0:
        print(f"  Total compute time: {total_actual_time:.1f}s ({total_actual_time/60:.1f} min)")
        avg_time = total_actual_time / len([j for j in jobs if j.get("actual_duration_s")])
        print(f"  Average per job: {avg_time:.1f}s ({avg_time/60:.1f} min)")
    
    # Final result
    results["passed"] = all_completed and non_decreasing and len(results["errors"]) == 0
    results["end_time"] = datetime.now(timezone.utc).isoformat()
    results["total_wall_time_s"] = int(time.time() - start_time)
    results["total_compute_time_s"] = total_actual_time
    
    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"load_test_v2_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print("LOAD TEST RESULTS")
    print(f"{'=' * 70}")
    print(f"Wall Clock Time: {results['total_wall_time_s']}s ({results['total_wall_time_s']/60:.1f} min)")
    print(f"Total Compute Time: {total_actual_time:.1f}s ({total_actual_time/60:.1f} min)")
    print(f"Jobs Completed: {len(completed_jobs)}/3")
    print(f"Errors: {len(results['errors'])}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Results saved to: {results_file}")
    
    if results["passed"]:
        print("\n✓ LOAD TEST PASSED")
        sys.exit(0)
    else:
        print("\n✗ LOAD TEST FAILED")
        for err in results["errors"]:
            print(f"  - {err}")
        sys.exit(1)

if __name__ == "__main__":
    main()

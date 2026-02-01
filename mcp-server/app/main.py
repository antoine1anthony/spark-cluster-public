"""
Unified MCP Server for Spark Cluster Tools (Level 2 + A2A)
==========================================================

This server provides Model Context Protocol (MCP) access to all cluster tool servers:
- Weather Tool Server (FourCastNet + CorrDiff)
- Media Toolkit (Whisper, OCR, Prover)
- Knowledge Base
- PDF Ingestion
- A2A Gateway (Agent2Agent Protocol for Utility Belt collaboration)

Features:
- Context-based logging (visible in client UI)
- Progress reporting for long-running tasks
- Resources for document retrieval
- Dynamic prompt loading from filesystem
- A2A client tools for delegating tasks to specialized agents

MCP Protocol: Streamable HTTP Transport via FastMCP 2.x
A2A Protocol: JSON-RPC 2.0 over HTTP for agent collaboration
"""

import os
import json
import httpx
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP, Context

# Tool Server URLs - Configure via environment variables
WEATHER_URL = os.environ.get("WEATHER_URL", "http://localhost:6000")
MEDIA_URL = os.environ.get("MEDIA_URL", "http://localhost:5001")
KB_URL = os.environ.get("KB_URL", "http://localhost:8002")
PDF_URL = os.environ.get("PDF_URL", "http://localhost:8001")

# vLLM Prover URL (Utility Belt)
PROVER_URL = os.environ.get("PROVER_URL", "http://localhost:8005")

# A2A Gateway URL (Utility Belt)
A2A_GATEWAY_URL = os.environ.get("A2A_GATEWAY_URL", "http://localhost:9000")

# Auteur Video Worker URL (Utility Belt)
AUTEUR_URL = os.environ.get("AUTEUR_URL", "http://localhost:8000")
AUTEUR_TOKEN = os.environ.get("AUTEUR_TOKEN", "change-me-in-production")
AUTEUR_OUTPUT_DIR = Path(os.environ.get("AUTEUR_OUTPUT_DIR", "/app/auteur-output"))


def normalize_url(url: str) -> str:
    """Normalize a URL by stripping trailing slashes and ensuring http:// prefix."""
    url = url.strip()
    # Remove trailing slashes
    url = url.rstrip("/")
    # Add http:// if no protocol specified
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    return url


# Prompts directory (mounted in Docker)
PROMPTS_DIR = Path(os.environ.get("PROMPTS_DIR", "/app/prompts"))

# Create FastMCP Server
mcp = FastMCP("Spark Cluster Tools")

# HTTP client for proxying requests
_client: Optional[httpx.AsyncClient] = None


async def get_client() -> httpx.AsyncClient:
    """Get or create the HTTP client."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


# =============================================================================
# Weather Tools
# =============================================================================

@mcp.tool()
async def weather_forecast(
    region: str,
    hours_ahead: int = 24,
    include_derived: bool = True,
    ctx: Context = None
) -> str:
    """
    Get AI-powered weather forecast for a city using FourCastNet.
    
    Args:
        region: City ID. Options: pittsburg_ca, san_francisco, oakland, atlanta, 
                athens_ga, castries, rodney_bay, mountain_view_sl
        hours_ahead: Forecast horizon in hours (12, 24, 48, 72, or 168)
        include_derived: Include wind speed, heat index, pressure tendency
    
    Returns:
        JSON string with forecast data including temperature (C/K/F), 
        wind, humidity, and time series.
    """
    if ctx:
        await ctx.info(f"Fetching {hours_ahead}h forecast for {region}...")
    
    client = await get_client()
    try:
        response = await client.post(
            f"{WEATHER_URL}/forecast",
            json={
                "region": region,
                "hours_ahead": hours_ahead,
                "include_derived": include_derived,
                "return_time_series": True
            }
        )
        response.raise_for_status()
        if ctx:
            await ctx.info(f"Forecast retrieved successfully for {region}")
        return json.dumps(response.json(), indent=2)
    except httpx.HTTPError as e:
        error_msg = f"Weather forecast error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def weather_forecast_detailed(
    region: str,
    hours_ahead: int = 48,
    samples: int = 4,
    ctx: Context = None
) -> str:
    """
    Get high-resolution weather forecast using CorrDiff with ensemble uncertainty.
    Generates visualization tiles for UI display.
    
    Args:
        region: City ID. Options: pittsburg_ca, san_francisco, oakland, atlanta, 
                athens_ga, castries, rodney_bay, mountain_view_sl
        hours_ahead: Forecast horizon in hours (12, 24, 48, 72, or 168)
        samples: Number of ensemble samples for uncertainty quantification (1-10)
    
    Returns:
        JSON string with high-res forecast data, ensemble statistics, 
        and URLs to generated map tiles.
    """
    if ctx:
        await ctx.info(f"Starting detailed CorrDiff forecast for {region} ({samples} samples)...")
        await ctx.report_progress(0, 100)
    
    client = await get_client()
    try:
        if ctx:
            await ctx.info("Running FourCastNet coarse prediction...")
            await ctx.report_progress(20, 100)
        
        response = await client.post(
            f"{WEATHER_URL}/forecast/detailed",
            json={
                "region": region,
                "hours_ahead": hours_ahead,
                "samples": samples
            }
        )
        
        if ctx:
            await ctx.info("Processing ensemble results...")
            await ctx.report_progress(80, 100)
        
        response.raise_for_status()
        
        if ctx:
            await ctx.info(f"Detailed forecast complete for {region}")
            await ctx.report_progress(100, 100)
        
        return json.dumps(response.json(), indent=2)
    except httpx.HTTPError as e:
        error_msg = f"Detailed forecast error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e)})


# =============================================================================
# Media Tools
# =============================================================================

@mcp.tool()
async def transcribe(
    audio_url: str,
    language: str = "en",
    ctx: Context = None
) -> str:
    """
    Transcribe audio to text using Whisper large-v3-turbo.
    
    Args:
        audio_url: URL to the audio file (mp3, wav, m4a, etc.)
        language: Language code for transcription (e.g., 'en', 'es', 'fr')
    
    Returns:
        JSON string with transcription text and metadata.
    """
    if ctx:
        await ctx.info(f"Starting transcription (language: {language})...")
        await ctx.report_progress(0, 100)
    
    client = await get_client()
    try:
        if ctx:
            await ctx.info("Downloading and processing audio...")
            await ctx.report_progress(20, 100)
        
        response = await client.post(
            f"{MEDIA_URL}/transcribe",
            json={"audio_url": audio_url, "language": language}
        )
        
        if ctx:
            await ctx.info("Running Whisper inference...")
            await ctx.report_progress(70, 100)
        
        response.raise_for_status()
        
        if ctx:
            await ctx.info("Transcription complete")
            await ctx.report_progress(100, 100)
        
        return json.dumps(response.json(), indent=2)
    except httpx.HTTPError as e:
        error_msg = f"Transcribe error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def ocr(
    image_url: str,
    ctx: Context = None
) -> str:
    """
    Extract text from images using OCR.
    
    Args:
        image_url: URL to the image file (png, jpg, etc.)
    
    Returns:
        JSON string with extracted text and bounding boxes.
    """
    if ctx:
        await ctx.info("Processing image for OCR...")
    
    client = await get_client()
    try:
        response = await client.post(
            f"{MEDIA_URL}/ocr",
            json={"image_url": image_url}
        )
        response.raise_for_status()
        if ctx:
            await ctx.info("OCR extraction complete")
        return json.dumps(response.json(), indent=2)
    except httpx.HTTPError as e:
        error_msg = f"OCR error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def prove(
    statement: str,
    proof_context: str = "",
    ctx: Context = None
) -> str:
    """
    Generate mathematical proofs using a reasoning model via vLLM.
    
    Args:
        statement: The mathematical statement or theorem to prove
        proof_context: Additional context, axioms, or definitions to use
    
    Returns:
        JSON string with the proof steps and verification status.
    """
    if ctx:
        await ctx.info("Analyzing mathematical statement...")
        await ctx.report_progress(0, 100)
    
    # Build the prompt for the prover
    system_prompt = "You are a mathematical proof assistant. Provide rigorous, step-by-step mathematical reasoning and proofs."
    
    user_content = f"Prove the following:\n\n{statement}"
    if proof_context:
        user_content = f"Context: {proof_context}\n\n{user_content}"
    
    client = await get_client()
    try:
        if ctx:
            await ctx.info("Generating proof...")
            await ctx.report_progress(30, 100)
        
        # Call vLLM chat completions endpoint
        response = await client.post(
            f"{PROVER_URL}/v1/chat/completions",
            json={
                "model": os.environ.get("PROVER_MODEL_NAME", "prover"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.1,
                "max_tokens": 4096
            }
        )
        response.raise_for_status()
        
        if ctx:
            await ctx.report_progress(90, 100)
        
        result = response.json()
        
        # Extract the proof from the response
        proof_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if ctx:
            await ctx.info("Proof generation complete")
            await ctx.report_progress(100, 100)
        
        return json.dumps({
            "statement": statement,
            "context": proof_context,
            "proof": proof_text,
            "status": "completed"
        }, indent=2)
        
    except httpx.HTTPError as e:
        error_msg = f"Prove error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e), "status": "failed"})
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e), "status": "failed"})


# =============================================================================
# Knowledge Base Tools
# =============================================================================

@mcp.tool()
async def kb_search(
    query: str,
    limit: int = 5,
    topic: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    Semantic search the knowledge base using pgvector embeddings.
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return (1-20)
        topic: Optional topic filter to narrow results
    
    Returns:
        JSON string with matching documents, scores, and snippets.
    """
    if ctx:
        await ctx.info(f"Searching knowledge base: '{query[:50]}...'")
    
    client = await get_client()
    try:
        payload = {"query": query, "limit": limit}
        if topic:
            payload["topic"] = topic
        
        response = await client.post(
            f"{KB_URL}/kb/search",
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        if ctx:
            count = len(result.get("results", []))
            await ctx.info(f"Found {count} matching documents")
        return json.dumps(result, indent=2)
    except httpx.HTTPError as e:
        error_msg = f"KB search error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e)})


# =============================================================================
# Knowledge Base Resources
# =============================================================================

@mcp.resource("kb://document/{document_id}")
async def get_kb_document(document_id: str) -> str:
    """
    Retrieve a specific document from the knowledge base by ID.
    
    This is exposed as an MCP Resource for semantic document access.
    """
    client = await get_client()
    try:
        response = await client.get(f"{KB_URL}/kb/document/{document_id}")
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except httpx.HTTPError as e:
        return json.dumps({"error": str(e)})


# Keep as tool for backward compatibility
@mcp.tool()
async def kb_document(
    document_id: str,
    ctx: Context = None
) -> str:
    """
    Retrieve a specific document from the knowledge base by ID.
    
    Args:
        document_id: Unique identifier of the document
    
    Returns:
        JSON string with the full document content and metadata.
    """
    if ctx:
        await ctx.info(f"Retrieving document: {document_id}")
    
    result = await get_kb_document(document_id)
    
    if ctx:
        await ctx.info("Document retrieved")
    
    return result


# =============================================================================
# PDF Ingestion Tools
# =============================================================================

@mcp.tool()
async def pdf_sync(
    folder_id: str = "",
    ctx: Context = None
) -> str:
    """
    Sync documents from Google Drive into the knowledge base.
    
    Args:
        folder_id: Google Drive folder ID to sync from (optional)
    
    Returns:
        JSON string with sync status and number of documents processed.
    """
    if ctx:
        await ctx.info("Starting Google Drive sync...")
        await ctx.report_progress(0, 100)
    
    client = await get_client()
    try:
        if ctx:
            await ctx.info("Connecting to Google Drive...")
            await ctx.report_progress(10, 100)
        
        response = await client.post(
            f"{PDF_URL}/sync/drive",
            json={"folder_id": folder_id} if folder_id else {}
        )
        
        if ctx:
            await ctx.info("Processing documents...")
            await ctx.report_progress(50, 100)
        
        response.raise_for_status()
        result = response.json()
        
        if ctx:
            count = result.get("documents_processed", 0)
            await ctx.info(f"Sync complete: {count} documents processed")
            await ctx.report_progress(100, 100)
        
        return json.dumps(result, indent=2)
    except httpx.HTTPError as e:
        error_msg = f"PDF sync error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e)})


# =============================================================================
# A2A Client Tools (Agent2Agent Protocol)
# =============================================================================

@mcp.tool()
async def a2a_discover_agents(
    url: str = "",
    ctx: Context = None
) -> str:
    """
    Discover available A2A agents and their capabilities.
    
    Fetches the Agent Card from an A2A-compliant server to understand
    what skills and capabilities are available for collaboration.
    
    Args:
        url: Base URL of the A2A Gateway (default: from environment)
    
    Returns:
        JSON string with the Agent Card containing skills, input/output modes,
        and capabilities.
    """
    gateway_url = url or A2A_GATEWAY_URL
    if ctx:
        await ctx.info(f"Discovering A2A agents at {gateway_url}...")
    
    client = await get_client()
    try:
        response = await client.get(
            f"{gateway_url}/.well-known/agent-card.json",
            headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        if ctx:
            skills = result.get("skills", [])
            skill_names = [s.get("name", s.get("id", "unknown")) for s in skills]
            await ctx.info(f"Found {len(skills)} agent skills: {', '.join(skill_names)}")
        
        return json.dumps(result, indent=2)
    except httpx.HTTPError as e:
        error_msg = f"A2A discovery error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def a2a_send_message(
    content: str,
    skill_hint: str = "",
    url: str = "",
    context_id: str = "",
    ctx: Context = None
) -> str:
    """
    Send a task to an A2A agent and get the result.
    
    Use this to delegate work to specialized agents in the Utility Belt:
    - Math/proofs (skill: math_reasoning)
    - Document OCR (skill: visual_document_understanding)
    - Audio transcription (skill: audio_transcription)
    
    Args:
        content: The task description or question to send to the agent
        skill_hint: Optional hint to route to specific skill 
                   (e.g., "math_reasoning", "visual_document_understanding")
        url: Base URL of the A2A Gateway (default: from environment)
        context_id: Optional context ID for multi-turn conversations
    
    Returns:
        JSON string with the Task object containing status and artifacts.
        On success, check result.artifacts[0].parts[0].text for the answer.
    """
    import uuid
    
    gateway_url = url or A2A_GATEWAY_URL
    if ctx:
        await ctx.info(f"Sending task to A2A agent{f' (skill: {skill_hint})' if skill_hint else ''}...")
        await ctx.report_progress(0, 100)
    
    # Build message content
    message_content = content
    if skill_hint:
        # Prepend skill hint to help routing
        message_content = f"[SKILL:{skill_hint}] {content}"
    
    # Build JSON-RPC request
    message_id = str(uuid.uuid4())
    rpc_request = {
        "jsonrpc": "2.0",
        "id": message_id,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": message_id,
                "parts": [{"kind": "text", "text": message_content}]
            }
        }
    }
    
    if context_id:
        rpc_request["params"]["message"]["contextId"] = context_id
    
    client = await get_client()
    try:
        if ctx:
            await ctx.info("Awaiting agent response...")
            await ctx.report_progress(30, 100)
        
        response = await client.post(
            gateway_url,
            json=rpc_request,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        if ctx:
            await ctx.report_progress(90, 100)
        
        # Check for RPC error
        if "error" in result:
            error_msg = f"A2A RPC error: {result['error']}"
            if ctx:
                await ctx.error(error_msg)
            return json.dumps(result, indent=2)
        
        # Extract task result
        task = result.get("result", {})
        status = task.get("status", {}).get("state", "unknown")
        
        if ctx:
            if status == "completed":
                await ctx.info(f"Task completed successfully (ID: {task.get('id', 'N/A')})")
            elif status == "failed":
                await ctx.error(f"Task failed: {task.get('status', {}).get('message', 'Unknown error')}")
            else:
                await ctx.info(f"Task status: {status}")
            await ctx.report_progress(100, 100)
        
        return json.dumps(task, indent=2)
    except httpx.HTTPError as e:
        error_msg = f"A2A send message error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def a2a_get_task(
    task_id: str,
    url: str = "",
    ctx: Context = None
) -> str:
    """
    Get the current status of an A2A task.
    
    Use this to check on long-running tasks or retrieve results
    for tasks that were submitted earlier.
    
    Args:
        task_id: The unique identifier of the task to retrieve
        url: Base URL of the A2A Gateway (default: from environment)
    
    Returns:
        JSON string with the Task object containing current status and artifacts.
    """
    import uuid
    
    gateway_url = url or A2A_GATEWAY_URL
    if ctx:
        await ctx.info(f"Retrieving task status: {task_id[:8]}...")
    
    rpc_request = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tasks/get",
        "params": {"taskId": task_id}
    }
    
    client = await get_client()
    try:
        response = await client.post(
            gateway_url,
            json=rpc_request,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        if "error" in result:
            error_msg = f"A2A get task error: {result['error']}"
            if ctx:
                await ctx.error(error_msg)
            return json.dumps(result, indent=2)
        
        task = result.get("result", {})
        status = task.get("status", {}).get("state", "unknown")
        
        if ctx:
            await ctx.info(f"Task {task_id[:8]}... status: {status}")
        
        return json.dumps(task, indent=2)
    except httpx.HTTPError as e:
        error_msg = f"A2A get task error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def a2a_send_file(
    file_base64: str,
    file_name: str,
    media_type: str,
    instruction: str = "",
    url: str = "",
    ctx: Context = None
) -> str:
    """
    Send a file (image or audio) to an A2A agent for processing.
    
    Use this for:
    - OCR: Send images/PDFs to extract text (media_type: image/png, image/jpeg, application/pdf)
    - Transcription: Send audio to convert to text (media_type: audio/wav, audio/mp3)
    
    Args:
        file_base64: Base64-encoded file content
        file_name: Name of the file (e.g., "document.png", "recording.wav")
        media_type: MIME type of the file (e.g., "image/png", "audio/wav")
        instruction: Optional instruction to accompany the file
        url: Base URL of the A2A Gateway (default: from environment)
    
    Returns:
        JSON string with the Task object containing extracted text or transcription.
    """
    import uuid
    
    gateway_url = url or A2A_GATEWAY_URL
    if ctx:
        await ctx.info(f"Sending file to A2A agent: {file_name} ({media_type})...")
        await ctx.report_progress(0, 100)
    
    # Determine skill based on media type
    if media_type.startswith("audio/"):
        default_instruction = "Transcribe this audio file."
    elif media_type.startswith("image/") or media_type == "application/pdf":
        default_instruction = "Extract all text from this document."
    else:
        default_instruction = "Process this file."
    
    message_id = str(uuid.uuid4())
    rpc_request = {
        "jsonrpc": "2.0",
        "id": message_id,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": message_id,
                "parts": [
                    {"kind": "text", "text": instruction or default_instruction},
                    {
                        "kind": "file",
                        "file": {
                            "name": file_name,
                            "mediaType": media_type,
                            "fileWithBytes": file_base64
                        }
                    }
                ]
            }
        }
    }
    
    client = await get_client()
    try:
        if ctx:
            await ctx.info("Uploading and processing file...")
            await ctx.report_progress(30, 100)
        
        response = await client.post(
            gateway_url,
            json=rpc_request,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        if ctx:
            await ctx.report_progress(90, 100)
        
        if "error" in result:
            error_msg = f"A2A file processing error: {result['error']}"
            if ctx:
                await ctx.error(error_msg)
            return json.dumps(result, indent=2)
        
        task = result.get("result", {})
        status = task.get("status", {}).get("state", "unknown")
        
        if ctx:
            if status == "completed":
                await ctx.info(f"File processing completed")
            elif status == "failed":
                await ctx.error(f"File processing failed")
            else:
                await ctx.info(f"Task status: {status}")
            await ctx.report_progress(100, 100)
        
        return json.dumps(task, indent=2)
    except httpx.HTTPError as e:
        error_msg = f"A2A send file error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e)})


# =============================================================================
# Video Generation Tools (Auteur System)
# =============================================================================

@mcp.tool()
async def generate_video(
    prompt: str,
    model_tier: str = "workhorse",
    num_frames: int = 17,
    num_steps: int = 20,
    ctx: Context = None
) -> str:
    """
    Generate a video from a text prompt using the Auteur system.
    
    This uses HunyuanVideo (Text-to-Video) or SVD (Image-to-Video) models
    running on the GB10 GPU with unified memory architecture.
    
    Args:
        prompt: Text description of the video to generate
        model_tier: Model to use:
            - "workhorse" (default): HunyuanVideo T2V - best quality
            - "experimental": CogVideoX T2V - faster but less stable
        num_frames: Number of frames (default: 17 for ~2s video at 8fps)
        num_steps: Inference steps (default: 20, higher = better quality)
    
    Returns:
        JSON string with job_id, status, video_path, and ffprobe stats when complete.
        Video files are saved to the auteur-output directory.
    
    Example:
        generate_video("A cinematic drone shot of mountains at golden hour sunset")
    """
    if ctx:
        await ctx.info(f"Submitting video generation job ({model_tier})...")
        await ctx.report_progress(0, 100)
    
    client = await get_client()
    base_url = normalize_url(AUTEUR_URL)
    headers = {"Authorization": f"Bearer {AUTEUR_TOKEN}"}
    
    try:
        # Submit job
        response = await client.post(
            f"{base_url}/jobs/json",
            headers=headers,
            json={
                "model_tier": model_tier,
                "prompt": prompt,
                "num_frames": num_frames,
                "num_steps": num_steps
            },
            timeout=30.0
        )
        response.raise_for_status()
        job_result = response.json()
        job_id = job_result["job_id"]
        
        if ctx:
            await ctx.info(f"Job submitted: {job_id}")
            await ctx.report_progress(10, 100)
        
        # Poll for completion (video generation takes 2-10 minutes)
        max_polls = 120  # 10 minutes max
        poll_interval = 5
        ffprobe_stats = None
        
        for poll_num in range(max_polls):
            await asyncio.sleep(poll_interval)
            
            status_response = await client.get(
                f"{base_url}/jobs/{job_id}",
                headers=headers,
                timeout=10.0
            )
            status_response.raise_for_status()
            job_status = status_response.json()
            
            progress = job_status.get("progress", 0)
            status = job_status.get("status", "unknown")
            
            if ctx:
                scaled_progress = 10 + int(progress * 0.8)
                await ctx.report_progress(scaled_progress, 100)
            
            if status == "completed":
                ffprobe_stats = job_status.get("ffprobe_summary")
                
                if ctx:
                    await ctx.info("Video generation complete! Downloading...")
                    if ffprobe_stats:
                        await ctx.info(
                            f"Video stats: {ffprobe_stats.get('width')}x{ffprobe_stats.get('height')}, "
                            f"{ffprobe_stats.get('nb_frames')} frames, "
                            f"{ffprobe_stats.get('size_bytes', 0) / 1024:.1f}KB"
                        )
                    await ctx.report_progress(90, 100)
                
                # Download the video to local output directory
                video_response = await client.get(
                    f"{base_url}/jobs/{job_id}/download",
                    headers=headers,
                    timeout=120.0
                )
                video_response.raise_for_status()
                
                AUTEUR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                local_video_path = AUTEUR_OUTPUT_DIR / f"auteur_{job_id}.mp4"
                with open(local_video_path, "wb") as f:
                    f.write(video_response.content)
                
                if ctx:
                    await ctx.info(f"Video saved: {local_video_path}")
                    await ctx.report_progress(100, 100)
                
                result = {
                    "status": "completed",
                    "job_id": job_id,
                    "prompt": prompt,
                    "model_tier": model_tier,
                    "num_frames": num_frames,
                    "video_path": str(local_video_path),
                    "message": f"Video generated successfully. File: {local_video_path}"
                }
                
                if ffprobe_stats:
                    result["ffprobe"] = {
                        "width": ffprobe_stats.get("width"),
                        "height": ffprobe_stats.get("height"),
                        "frames": ffprobe_stats.get("nb_frames"),
                        "duration_s": ffprobe_stats.get("duration"),
                        "size_bytes": ffprobe_stats.get("size_bytes"),
                        "codec": ffprobe_stats.get("codec")
                    }
                
                return json.dumps(result, indent=2)
                
            elif status == "failed":
                error = job_status.get("error", "Unknown error")
                if ctx:
                    await ctx.error(f"Video generation failed: {error}")
                return json.dumps({
                    "status": "failed",
                    "job_id": job_id,
                    "error": error
                }, indent=2)
            
            if ctx and poll_num % 6 == 0:
                await ctx.info(f"Generating video... {progress}%")
        
        if ctx:
            await ctx.error("Video generation timed out")
        return json.dumps({
            "status": "timeout",
            "job_id": job_id,
            "message": "Video generation timed out after 10 minutes."
        }, indent=2)
        
    except httpx.HTTPError as e:
        error_msg = f"Video generation error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e), "status": "failed"})


@mcp.tool()
async def generate_video_from_image(
    image_path: str,
    prompt: str = "animate this image",
    num_frames: int = 25,
    num_steps: int = 25,
    ctx: Context = None
) -> str:
    """
    Generate a video from an input image using SVD (Stable Video Diffusion).
    
    This creates smooth motion from a still image, turning photos into
    short video clips with natural movement.
    
    Args:
        image_path: Path to the input image file (jpg, png)
        prompt: Optional description of desired motion/animation
        num_frames: Number of frames (default: 25 for ~1.5s video)
        num_steps: Inference steps (default: 25)
    
    Returns:
        JSON string with job_id, status, and video_path when complete.
    
    Example:
        generate_video_from_image("/path/to/mountain.jpg", "gentle camera motion")
    """
    import base64
    
    if ctx:
        await ctx.info(f"Loading image and submitting I2V job...")
        await ctx.report_progress(0, 100)
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        error_msg = f"Image file not found: {image_path}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": error_msg, "status": "failed"})
    except Exception as e:
        error_msg = f"Failed to read image: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": error_msg, "status": "failed"})
    
    client = await get_client()
    base_url = normalize_url(AUTEUR_URL)
    headers = {"Authorization": f"Bearer {AUTEUR_TOKEN}"}
    
    try:
        files = {"input_image": (Path(image_path).name, image_bytes, "image/jpeg")}
        data = {
            "model_tier": "validator",
            "prompt": prompt,
            "num_frames": str(num_frames),
            "num_steps": str(num_steps)
        }
        
        response = await client.post(
            f"{base_url}/jobs",
            headers=headers,
            files=files,
            data=data,
            timeout=30.0
        )
        response.raise_for_status()
        job_result = response.json()
        job_id = job_result["job_id"]
        
        if ctx:
            await ctx.info(f"I2V job submitted: {job_id}")
            await ctx.report_progress(10, 100)
        
        max_polls = 60
        poll_interval = 5
        ffprobe_stats = None
        
        for poll_num in range(max_polls):
            await asyncio.sleep(poll_interval)
            
            status_response = await client.get(
                f"{base_url}/jobs/{job_id}",
                headers=headers,
                timeout=10.0
            )
            status_response.raise_for_status()
            job_status = status_response.json()
            
            progress = job_status.get("progress", 0)
            status = job_status.get("status", "unknown")
            
            if ctx:
                scaled_progress = 10 + int(progress * 0.8)
                await ctx.report_progress(scaled_progress, 100)
            
            if status == "completed":
                ffprobe_stats = job_status.get("ffprobe_summary")
                
                if ctx:
                    await ctx.info("I2V generation complete! Downloading...")
                    if ffprobe_stats:
                        await ctx.info(
                            f"Video stats: {ffprobe_stats.get('width')}x{ffprobe_stats.get('height')}, "
                            f"{ffprobe_stats.get('nb_frames')} frames"
                        )
                    await ctx.report_progress(90, 100)
                
                video_response = await client.get(
                    f"{base_url}/jobs/{job_id}/download",
                    headers=headers,
                    timeout=120.0
                )
                video_response.raise_for_status()
                
                AUTEUR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                local_video_path = AUTEUR_OUTPUT_DIR / f"auteur_i2v_{job_id}.mp4"
                with open(local_video_path, "wb") as f:
                    f.write(video_response.content)
                
                if ctx:
                    await ctx.info(f"Video saved: {local_video_path}")
                    await ctx.report_progress(100, 100)
                
                result = {
                    "status": "completed",
                    "job_id": job_id,
                    "input_image": image_path,
                    "prompt": prompt,
                    "model_tier": "validator (SVD)",
                    "num_frames": num_frames,
                    "video_path": str(local_video_path),
                    "message": f"Video generated from image. File: {local_video_path}"
                }
                
                if ffprobe_stats:
                    result["ffprobe"] = {
                        "width": ffprobe_stats.get("width"),
                        "height": ffprobe_stats.get("height"),
                        "frames": ffprobe_stats.get("nb_frames"),
                        "duration_s": ffprobe_stats.get("duration"),
                        "size_bytes": ffprobe_stats.get("size_bytes"),
                        "codec": ffprobe_stats.get("codec")
                    }
                
                return json.dumps(result, indent=2)
                
            elif status == "failed":
                error = job_status.get("error", "Unknown error")
                if ctx:
                    await ctx.error(f"I2V generation failed: {error}")
                return json.dumps({
                    "status": "failed",
                    "job_id": job_id,
                    "error": error
                }, indent=2)
            
            if ctx and poll_num % 6 == 0:
                await ctx.info(f"Generating video from image... {progress}%")
        
        if ctx:
            await ctx.error("I2V generation timed out")
        return json.dumps({
            "status": "timeout",
            "job_id": job_id,
            "message": "Video generation timed out."
        }, indent=2)
        
    except httpx.HTTPError as e:
        error_msg = f"I2V generation error: {e}"
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({"error": str(e), "status": "failed"})


# Import asyncio for async operations
import asyncio


# =============================================================================
# Dynamic Prompt Loading
# =============================================================================

def load_prompts_from_directory():
    """
    Load all agent prompts from the prompts directory.
    
    Walks the directory structure and registers each .md file as an MCP prompt.
    Prompt names are derived from the file path (e.g., coder/full_stack_developer).
    """
    if not PROMPTS_DIR.exists():
        print(f"Prompts directory not found: {PROMPTS_DIR}")
        return
    
    prompt_count = 0
    
    for md_file in PROMPTS_DIR.rglob("*.md"):
        # Skip README files
        if md_file.name.lower() == "readme.md":
            continue
        
        # Build prompt name from relative path (e.g., coder/full_stack_developer)
        rel_path = md_file.relative_to(PROMPTS_DIR)
        prompt_name = str(rel_path.with_suffix("")).replace("/", "_").replace("\\", "_")
        
        # Read prompt content
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Failed to read {md_file}: {e}")
            continue
        
        # Extract title from first line if it starts with #
        lines = content.strip().split("\n")
        description = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else prompt_name
        
        # Register the prompt dynamically
        # We create a closure to capture the content
        def make_prompt_func(prompt_content: str, prompt_desc: str):
            async def prompt_func() -> str:
                return prompt_content
            prompt_func.__doc__ = prompt_desc
            return prompt_func
        
        # Register with FastMCP
        prompt_func = make_prompt_func(content, description)
        mcp.prompt(name=prompt_name)(prompt_func)
        prompt_count += 1
        print(f"Registered prompt: {prompt_name}")
    
    print(f"Loaded {prompt_count} prompts from {PROMPTS_DIR}")


# =============================================================================
# Entry Point
# =============================================================================

def run_server():
    """Run the MCP server with streamable-http transport."""
    # Load prompts from filesystem
    load_prompts_from_directory()
    
    # Get port from environment or default
    port = int(os.environ.get("MCP_PORT", "5050"))
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    
    print(f"Starting FastMCP server on {host}:{port} with streamable-http transport")
    mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    run_server()

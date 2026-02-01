#!/usr/bin/env python3
"""
A2A Gateway for Spark Utility Belt

This service wraps the utility models (Prover, OCR, Whisper, Auteur) as discoverable
A2A agents, enabling AI orchestration systems to collaborate with them using
the Agent2Agent protocol.

All configuration is done via environment variables for easy deployment customization.
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - All services use environment variables
PROVER_URL = os.getenv("PROVER_URL", "http://prover:8005")
OCR_URL = os.getenv("OCR_URL", "http://ocr:8000")
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:8000")
AUTEUR_URL = os.getenv("AUTEUR_URL", "http://auteur-worker:8000")
AUTEUR_TOKEN = os.getenv("AUTEUR_TOKEN", "change-me-in-production")

# Gateway host advertised in Agent Card
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "9000"))

app = FastAPI(title="Spark Utility Belt - A2A Gateway", version="0.3.0")


# ============================================================================
# A2A Protocol Data Models
# ============================================================================

class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = []
    examples: list[str] = []
    inputModes: list[str] = ["text/plain"]
    outputModes: list[str] = ["text/plain"]


class AgentCapabilities(BaseModel):
    streaming: bool = True
    pushNotifications: bool = False


class AgentCard(BaseModel):
    name: str
    description: str
    url: str
    version: str = "0.3.0"
    capabilities: AgentCapabilities = AgentCapabilities()
    defaultInputModes: list[str] = ["text/plain"]
    defaultOutputModes: list[str] = ["text/plain"]
    skills: list[AgentSkill] = []


class TextPart(BaseModel):
    kind: str = "text"
    text: str


class FilePart(BaseModel):
    kind: str = "file"
    file: dict  # {name, mediaType, fileWithBytes or fileWithUri}


class DataPart(BaseModel):
    kind: str = "data"
    data: dict


class Message(BaseModel):
    role: str  # "user" or "agent"
    messageId: str
    parts: list[dict]
    contextId: Optional[str] = None
    referenceTaskIds: Optional[list[str]] = None


class TaskStatus(BaseModel):
    state: TaskState
    message: Optional[Message] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class Artifact(BaseModel):
    artifactId: str
    name: str
    description: Optional[str] = None
    parts: list[dict]


class Task(BaseModel):
    id: str
    contextId: str
    status: TaskStatus
    artifacts: list[Artifact] = []


class SendMessageRequest(BaseModel):
    message: Message
    metadata: Optional[dict] = None


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int
    method: str
    params: Optional[dict] = None


class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int
    result: Optional[Any] = None
    error: Optional[dict] = None


# ============================================================================
# Agent Card Definition
# ============================================================================

AGENT_CARD = AgentCard(
    name="Spark Utility Belt",
    description="A collection of specialized AI agents for mathematical reasoning, document processing, audio transcription, and video generation.",
    url=f"http://{GATEWAY_HOST}:{GATEWAY_PORT}",
    version="0.3.0",
    capabilities=AgentCapabilities(streaming=True, pushNotifications=False),
    defaultInputModes=["text/plain", "application/json"],
    defaultOutputModes=["text/plain", "application/json"],
    skills=[
        AgentSkill(
            id="math_reasoning",
            name="Mathematical Reasoning",
            description="Solve complex mathematical problems, prove theorems, and perform logical deductions.",
            tags=["math", "logic", "proof", "theorem"],
            examples=[
                "Prove that sqrt(2) is irrational",
                "Solve the differential equation dy/dx = y",
                "Verify this mathematical proof step by step"
            ],
            inputModes=["text/plain"],
            outputModes=["text/plain"]
        ),
        AgentSkill(
            id="visual_document_understanding",
            name="Document OCR & Understanding",
            description="Extract and understand text from images, PDFs, and scanned documents.",
            tags=["ocr", "document", "text-extraction", "vision"],
            examples=[
                "Extract text from this image",
                "What does this document say?",
                "Read the text in this scanned PDF"
            ],
            inputModes=["image/png", "image/jpeg", "application/pdf", "text/plain"],
            outputModes=["text/plain", "application/json"]
        ),
        AgentSkill(
            id="audio_transcription",
            name="Audio Transcription",
            description="Transcribe speech from audio files to text using Whisper.",
            tags=["audio", "speech", "transcription", "whisper"],
            examples=[
                "Transcribe this audio file",
                "Convert this speech to text",
                "What is being said in this recording?"
            ],
            inputModes=["audio/wav", "audio/mp3", "audio/mpeg", "audio/webm"],
            outputModes=["text/plain", "application/json"]
        ),
        AgentSkill(
            id="video_generation",
            name="AI Video Generation",
            description="Generate videos from text prompts (T2V) or images (I2V) using HunyuanVideo and SVD models.",
            tags=["video", "generation", "ai", "creative", "hunyuan", "svd"],
            examples=[
                "Generate a video of a sunset over mountains",
                "Create a cinematic video of waves crashing on rocks",
                "Animate this image into a video"
            ],
            inputModes=["text/plain", "image/png", "image/jpeg"],
            outputModes=["video/mp4", "application/json"]
        )
    ]
)


# ============================================================================
# In-Memory Task Store
# ============================================================================

tasks: dict[str, Task] = {}
contexts: dict[str, list[str]] = {}  # contextId -> [taskIds]


# ============================================================================
# Skill Handlers
# ============================================================================

async def handle_math_reasoning(message: Message) -> Task:
    """Route to reasoning model for mathematical reasoning."""
    task_id = str(uuid.uuid4())
    context_id = message.contextId or str(uuid.uuid4())
    
    # Extract text content from message
    text_content = ""
    for part in message.parts:
        if part.get("kind") == "text" or "text" in part:
            text_content += part.get("text", "")
    
    # Create task in working state
    task = Task(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(state=TaskState.WORKING)
    )
    tasks[task_id] = task
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{PROVER_URL}/v1/chat/completions",
                json={
                    "model": os.getenv("PROVER_MODEL_NAME", "prover"),
                    "messages": [
                        {"role": "system", "content": "You are a mathematical proof assistant. Provide rigorous, step-by-step mathematical reasoning."},
                        {"role": "user", "content": text_content}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096
                }
            )
            response.raise_for_status()
            result = response.json()
            
            answer = result["choices"][0]["message"]["content"]
            
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.artifacts = [
                Artifact(
                    artifactId=str(uuid.uuid4()),
                    name="mathematical_proof",
                    description="Mathematical reasoning result",
                    parts=[{"kind": "text", "text": answer}]
                )
            ]
    except Exception as e:
        logger.error(f"Math reasoning failed: {e}")
        task.status = TaskStatus(
            state=TaskState.FAILED,
            message=Message(
                role="agent",
                messageId=str(uuid.uuid4()),
                parts=[{"kind": "text", "text": f"Error: {str(e)}"}]
            )
        )
    
    tasks[task_id] = task
    return task


async def handle_document_ocr(message: Message) -> Task:
    """Route to OCR model for document understanding."""
    task_id = str(uuid.uuid4())
    context_id = message.contextId or str(uuid.uuid4())
    
    task = Task(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(state=TaskState.WORKING)
    )
    tasks[task_id] = task
    
    # Extract content - could be text instruction or file
    text_content = ""
    image_data = None
    
    for part in message.parts:
        if part.get("kind") == "text" or "text" in part:
            text_content += part.get("text", "")
        elif part.get("kind") == "file":
            file_info = part.get("file", {})
            if "fileWithBytes" in file_info:
                image_data = file_info["fileWithBytes"]
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            messages = [{"role": "system", "content": "You are a document OCR specialist. Extract and understand text from images accurately."}]
            
            if image_data:
                # Image with base64 data
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_content or "Extract all text from this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                })
            else:
                messages.append({"role": "user", "content": text_content})
            
            response = await client.post(
                f"{OCR_URL}/v1/chat/completions",
                json={
                    "model": os.getenv("OCR_MODEL_NAME", "ocr"),
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 4096
                }
            )
            response.raise_for_status()
            result = response.json()
            
            answer = result["choices"][0]["message"]["content"]
            
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.artifacts = [
                Artifact(
                    artifactId=str(uuid.uuid4()),
                    name="extracted_text",
                    description="OCR extraction result",
                    parts=[{"kind": "text", "text": answer}]
                )
            ]
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        task.status = TaskStatus(
            state=TaskState.FAILED,
            message=Message(
                role="agent",
                messageId=str(uuid.uuid4()),
                parts=[{"kind": "text", "text": f"Error: {str(e)}"}]
            )
        )
    
    tasks[task_id] = task
    return task


async def handle_audio_transcription(message: Message) -> Task:
    """Route to Whisper for audio transcription."""
    task_id = str(uuid.uuid4())
    context_id = message.contextId or str(uuid.uuid4())
    
    task = Task(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(state=TaskState.WORKING)
    )
    tasks[task_id] = task
    
    # Extract audio file from message
    audio_data = None
    audio_name = "audio.wav"
    
    for part in message.parts:
        if part.get("kind") == "file":
            file_info = part.get("file", {})
            if "fileWithBytes" in file_info:
                audio_data = base64.b64decode(file_info["fileWithBytes"])
                audio_name = file_info.get("name", "audio.wav")
    
    if not audio_data:
        task.status = TaskStatus(
            state=TaskState.FAILED,
            message=Message(
                role="agent",
                messageId=str(uuid.uuid4()),
                parts=[{"kind": "text", "text": "Error: No audio file provided"}]
            )
        )
        tasks[task_id] = task
        return task
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Whisper expects multipart form data
            files = {"file": (audio_name, audio_data)}
            response = await client.post(
                f"{WHISPER_URL}/v1/audio/transcriptions",
                files=files,
                data={"model": os.getenv("WHISPER_MODEL_NAME", "whisper-large-v3-turbo")}
            )
            response.raise_for_status()
            result = response.json()
            
            transcription = result.get("text", "")
            
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.artifacts = [
                Artifact(
                    artifactId=str(uuid.uuid4()),
                    name="transcription",
                    description="Audio transcription result",
                    parts=[{"kind": "text", "text": transcription}]
                )
            ]
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        task.status = TaskStatus(
            state=TaskState.FAILED,
            message=Message(
                role="agent",
                messageId=str(uuid.uuid4()),
                parts=[{"kind": "text", "text": f"Error: {str(e)}"}]
            )
        )
    
    tasks[task_id] = task
    return task


async def handle_video_generation(message: Message) -> Task:
    """Route to Auteur Worker for video generation."""
    task_id = str(uuid.uuid4())
    context_id = message.contextId or str(uuid.uuid4())
    
    task = Task(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(state=TaskState.WORKING)
    )
    tasks[task_id] = task
    
    # Extract content - text prompt or image for I2V
    text_content = ""
    image_data = None
    image_name = "input.jpg"
    
    for part in message.parts:
        if part.get("kind") == "text" or "text" in part:
            text_content += part.get("text", "")
        elif part.get("kind") == "file":
            file_info = part.get("file", {})
            if "fileWithBytes" in file_info:
                media_type = file_info.get("mediaType", "")
                if media_type.startswith("image/"):
                    image_data = base64.b64decode(file_info["fileWithBytes"])
                    image_name = file_info.get("name", "input.jpg")
    
    try:
        headers = {"Authorization": f"Bearer {AUTEUR_TOKEN}"}
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            # Determine model tier based on input
            if image_data:
                # I2V mode - use validator (SVD)
                files = {"input_image": (image_name, image_data)}
                data = {
                    "model_tier": "validator",
                    "prompt": text_content or "animate this image",
                    "num_frames": 25,
                    "num_steps": 25
                }
                response = await client.post(
                    f"{AUTEUR_URL}/jobs",
                    headers=headers,
                    files=files,
                    data=data
                )
            else:
                # T2V mode - use workhorse (HunyuanVideo)
                response = await client.post(
                    f"{AUTEUR_URL}/jobs/json",
                    headers=headers,
                    json={
                        "model_tier": "workhorse",
                        "prompt": text_content,
                        "num_frames": 17,
                        "num_steps": 20
                    }
                )
            
            response.raise_for_status()
            job_result = response.json()
            job_id = job_result["job_id"]
            
            logger.info(f"Video job submitted: {job_id}")
            
            # Poll for completion (video generation can take minutes)
            max_polls = 120  # 10 minutes max
            poll_interval = 5
            
            for _ in range(max_polls):
                await asyncio.sleep(poll_interval)
                
                status_response = await client.get(
                    f"{AUTEUR_URL}/jobs/{job_id}",
                    headers=headers
                )
                status_response.raise_for_status()
                job_status = status_response.json()
                
                if job_status["status"] == "completed":
                    # Download the video
                    video_response = await client.get(
                        f"{AUTEUR_URL}/jobs/{job_id}/download",
                        headers=headers
                    )
                    video_response.raise_for_status()
                    video_bytes = video_response.content
                    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
                    
                    task.status = TaskStatus(state=TaskState.COMPLETED)
                    task.artifacts = [
                        Artifact(
                            artifactId=str(uuid.uuid4()),
                            name="generated_video",
                            description=f"AI-generated video from prompt: {text_content[:50]}...",
                            parts=[
                                {"kind": "text", "text": f"Video generated successfully. Job ID: {job_id}"},
                                {
                                    "kind": "file",
                                    "file": {
                                        "name": f"auteur_{job_id}.mp4",
                                        "mediaType": "video/mp4",
                                        "fileWithBytes": video_b64
                                    }
                                }
                            ]
                        )
                    ]
                    break
                    
                elif job_status["status"] == "failed":
                    raise Exception(f"Video generation failed: {job_status.get('error', 'Unknown error')}")
                    
                # Still running, continue polling
                logger.info(f"Job {job_id} status: {job_status['status']}, progress: {job_status.get('progress', 0)}%")
            
            else:
                # Timeout
                raise Exception("Video generation timed out after 10 minutes")
                
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        task.status = TaskStatus(
            state=TaskState.FAILED,
            message=Message(
                role="agent",
                messageId=str(uuid.uuid4()),
                parts=[{"kind": "text", "text": f"Error: {str(e)}"}]
            )
        )
    
    tasks[task_id] = task
    return task


def detect_skill(message: Message, skill_hint: Optional[str] = None) -> str:
    """Detect which skill to use based on message content or explicit hint."""
    # If a skill hint is provided, use it directly
    if skill_hint:
        valid_skills = ["math_reasoning", "visual_document_understanding", 
                       "audio_transcription", "video_generation"]
        if skill_hint in valid_skills:
            return skill_hint
    
    text_content = ""
    has_image = False
    has_audio = False
    
    for part in message.parts:
        if part.get("kind") == "text" or "text" in part:
            text_content += part.get("text", "").lower()
        elif part.get("kind") == "file":
            file_info = part.get("file", {})
            media_type = file_info.get("mediaType", "")
            if media_type.startswith("image/") or media_type == "application/pdf":
                has_image = True
            elif media_type.startswith("audio/"):
                has_audio = True
    
    # Priority: explicit file types > text analysis
    if has_audio:
        return "audio_transcription"
    
    # Video generation keywords (check before OCR for images)
    video_keywords = ["generate video", "create video", "make video", "video of",
                     "animate", "video generation", "t2v", "i2v", "text to video",
                     "image to video", "cinematic", "footage"]
    if any(kw in text_content for kw in video_keywords):
        return "video_generation"
    
    # If image + video-like prompt, use video generation
    if has_image and any(kw in text_content for kw in ["animate", "video", "motion"]):
        return "video_generation"
    
    # OCR for images with text extraction intent
    if has_image:
        return "visual_document_understanding"
    
    # Text-based detection
    math_keywords = ["prove", "theorem", "equation", "derivative", "integral", 
                     "solve", "calculate", "math", "proof", "lemma", "sqrt"]
    ocr_keywords = ["ocr", "extract text", "read document", "scan"]
    
    if any(kw in text_content for kw in math_keywords):
        return "math_reasoning"
    if any(kw in text_content for kw in ocr_keywords):
        return "visual_document_understanding"
    
    # Default to math reasoning for text-only requests
    return "math_reasoning"


# ============================================================================
# A2A Protocol Endpoints
# ============================================================================

@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    """Agent Discovery endpoint."""
    return AGENT_CARD.model_dump()


@app.get("/.well-known/agent-card")
async def get_agent_card_alt():
    """Alternative Agent Discovery endpoint."""
    return AGENT_CARD.model_dump()


@app.post("/")
async def handle_jsonrpc(request: Request):
    """Main JSON-RPC endpoint for A2A protocol."""
    try:
        body = await request.json()
        rpc_request = JsonRpcRequest(**body)
        
        if rpc_request.method == "message/send":
            return await handle_message_send(rpc_request)
        elif rpc_request.method == "tasks/get":
            return await handle_tasks_get(rpc_request)
        elif rpc_request.method == "tasks/cancel":
            return await handle_tasks_cancel(rpc_request)
        else:
            return JsonRpcResponse(
                id=rpc_request.id,
                error={"code": -32601, "message": f"Method not found: {rpc_request.method}"}
            ).model_dump()
    except Exception as e:
        logger.error(f"JSON-RPC error: {e}")
        return JsonRpcResponse(
            id=body.get("id", 0) if isinstance(body, dict) else 0,
            error={"code": -32603, "message": str(e)}
        ).model_dump()


async def handle_message_send(rpc_request: JsonRpcRequest) -> dict:
    """Handle message/send method."""
    params = rpc_request.params or {}
    message_data = params.get("message", {})
    metadata = params.get("metadata", {})
    
    # Support skill_hint in metadata for explicit routing
    skill_hint = metadata.get("skill_hint") or metadata.get("skillHint")
    
    message = Message(
        role=message_data.get("role", "user"),
        messageId=message_data.get("messageId", str(uuid.uuid4())),
        parts=message_data.get("parts", []),
        contextId=message_data.get("contextId"),
        referenceTaskIds=message_data.get("referenceTaskIds")
    )
    
    # Detect skill and route
    skill_id = detect_skill(message, skill_hint)
    logger.info(f"Routing to skill: {skill_id}")
    
    if skill_id == "math_reasoning":
        task = await handle_math_reasoning(message)
    elif skill_id == "visual_document_understanding":
        task = await handle_document_ocr(message)
    elif skill_id == "audio_transcription":
        task = await handle_audio_transcription(message)
    elif skill_id == "video_generation":
        task = await handle_video_generation(message)
    else:
        task = await handle_math_reasoning(message)  # Default
    
    return JsonRpcResponse(
        id=rpc_request.id,
        result=task.model_dump()
    ).model_dump()


async def handle_tasks_get(rpc_request: JsonRpcRequest) -> dict:
    """Handle tasks/get method."""
    params = rpc_request.params or {}
    task_id = params.get("taskId")
    
    if not task_id or task_id not in tasks:
        return JsonRpcResponse(
            id=rpc_request.id,
            error={"code": -32602, "message": f"Task not found: {task_id}"}
        ).model_dump()
    
    return JsonRpcResponse(
        id=rpc_request.id,
        result=tasks[task_id].model_dump()
    ).model_dump()


async def handle_tasks_cancel(rpc_request: JsonRpcRequest) -> dict:
    """Handle tasks/cancel method."""
    params = rpc_request.params or {}
    task_id = params.get("taskId")
    
    if not task_id or task_id not in tasks:
        return JsonRpcResponse(
            id=rpc_request.id,
            error={"code": -32602, "message": f"Task not found: {task_id}"}
        ).model_dump()
    
    task = tasks[task_id]
    if task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
        return JsonRpcResponse(
            id=rpc_request.id,
            error={"code": -32602, "message": "Task already in terminal state"}
        ).model_dump()
    
    task.status = TaskStatus(state=TaskState.CANCELED)
    tasks[task_id] = task
    
    return JsonRpcResponse(
        id=rpc_request.id,
        result=task.model_dump()
    ).model_dump()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "a2a-gateway", "version": "0.3.0"}


@app.get("/agents")
async def list_agents():
    """List available agent skills."""
    return {"skills": [s.model_dump() for s in AGENT_CARD.skills]}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting A2A Gateway on port {GATEWAY_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)

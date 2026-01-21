#!/usr/bin/env python3
"""
Prompt Registry Service for Brain LLM

Provides endpoints to list and retrieve agent system prompts from markdown files.
Runs alongside vLLM to provide prompt management without modifying the model server.

Endpoints:
  GET /prompts - List all available prompts
  GET /prompts/{category}/{name} - Get a specific prompt
  GET /prompts/health - Health check

Placeholder Substitution:
  {{MODEL_NAME}} - Replaced with the active model name from vLLM
"""

import os
import re
import time
import urllib.request
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configuration - all via environment variables
PROMPTS_DIR = os.getenv("PROMPTS_DIR", "/app/agent-prompts")
PROMPT_PORT = int(os.getenv("PROMPT_PORT", "8010"))
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000")
MODEL_NAME_FALLBACK = os.getenv("MODEL_NAME", "brain-llm")

# Cache for model name (TTL-based)
_model_name_cache = {"name": None, "expires": 0}
MODEL_NAME_CACHE_TTL = 60  # seconds

app = FastAPI(
    title="Brain LLM Prompt Registry",
    description="Agent system prompt management for the Brain LLM",
    version="1.0.0"
)


def get_active_model_name() -> str:
    """
    Get the active model name from vLLM with caching.
    Falls back to MODEL_NAME env var if vLLM is unreachable.
    """
    global _model_name_cache
    
    now = time.time()
    if _model_name_cache["name"] and now < _model_name_cache["expires"]:
        return _model_name_cache["name"]
    
    try:
        req = urllib.request.Request(f"{VLLM_URL}/v1/models", method="GET")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if data.get("data") and len(data["data"]) > 0:
                model_name = data["data"][0].get("id", MODEL_NAME_FALLBACK)
                _model_name_cache = {"name": model_name, "expires": now + MODEL_NAME_CACHE_TTL}
                return model_name
    except Exception as e:
        print(f"Failed to fetch model name from vLLM: {e}")
    
    # Fallback to env var
    _model_name_cache = {"name": MODEL_NAME_FALLBACK, "expires": now + MODEL_NAME_CACHE_TTL}
    return MODEL_NAME_FALLBACK


def substitute_placeholders(content: str) -> str:
    """
    Replace placeholders in prompt content with runtime values.
    
    Supported placeholders:
      {{MODEL_NAME}} - Active model name from vLLM
    """
    model_name = get_active_model_name()
    content = content.replace("{{MODEL_NAME}}", model_name)
    return content


class PromptInfo(BaseModel):
    """Metadata about a prompt."""
    name: str
    category: str
    path: str
    title: Optional[str] = None
    description: Optional[str] = None


class PromptContent(BaseModel):
    """Full prompt content."""
    name: str
    category: str
    title: Optional[str] = None
    content: str
    role_summary: Optional[str] = None


def extract_title_and_role(content: str) -> tuple[Optional[str], Optional[str]]:
    """Extract title and role from markdown content."""
    title = None
    role = None
    
    # Extract title from first H1
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
    
    # Extract role section
    role_match = re.search(r'##\s+Role\s*\n+(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if role_match:
        role = role_match.group(1).strip()
    
    return title, role


def scan_prompts() -> dict[str, dict[str, PromptInfo]]:
    """Scan the prompts directory and build a registry."""
    registry = {}
    prompts_path = Path(PROMPTS_DIR)
    
    if not prompts_path.exists():
        return registry
    
    for category_dir in prompts_path.iterdir():
        if not category_dir.is_dir():
            continue
        if category_dir.name.startswith('.') or category_dir.name == '__pycache__':
            continue
            
        category = category_dir.name
        registry[category] = {}
        
        for prompt_file in category_dir.glob("*.md"):
            if prompt_file.name.startswith('README'):
                continue
                
            name = prompt_file.stem
            content = prompt_file.read_text()
            title, _ = extract_title_and_role(content)
            
            registry[category][name] = PromptInfo(
                name=name,
                category=category,
                path=str(prompt_file),
                title=title,
                description=f"{category}/{name}"
            )
    
    return registry


# Global registry - refreshed on each request for simplicity
# In production, could add caching with file watcher
@app.get("/prompts", response_model=dict)
def list_prompts():
    """List all available prompts organized by category."""
    registry = scan_prompts()
    result = {}
    for category, prompts in registry.items():
        result[category] = [
            {
                "name": p.name,
                "title": p.title,
                "path": f"/prompts/{category}/{p.name}"
            }
            for p in prompts.values()
        ]
    return {"prompts": result, "total": sum(len(v) for v in result.values())}


@app.get("/prompts/{category}/{name}", response_model=PromptContent)
def get_prompt(category: str, name: str):
    """Get a specific prompt by category and name."""
    prompts_path = Path(PROMPTS_DIR)
    prompt_file = prompts_path / category / f"{name}.md"
    
    if not prompt_file.exists():
        raise HTTPException(status_code=404, detail=f"Prompt {category}/{name} not found")
    
    content = prompt_file.read_text()
    content = substitute_placeholders(content)  # Apply runtime substitution
    title, role = extract_title_and_role(content)
    
    return PromptContent(
        name=name,
        category=category,
        title=title,
        content=content,
        role_summary=role
    )


@app.get("/prompts/health")
def health_check():
    """Health check endpoint."""
    prompts_path = Path(PROMPTS_DIR)
    return {
        "status": "ok",
        "prompts_dir": PROMPTS_DIR,
        "prompts_available": prompts_path.exists(),
        "active_model": get_active_model_name()
    }


@app.get("/prompts/system/{category}/{name}")
def get_system_prompt(category: str, name: str):
    """
    Get a prompt formatted as an OpenAI-compatible system message.
    
    Returns JSON that can be directly used in a chat completion request:
    {"role": "system", "content": "..."}
    
    Placeholders like {{MODEL_NAME}} are replaced with runtime values.
    """
    prompts_path = Path(PROMPTS_DIR)
    prompt_file = prompts_path / category / f"{name}.md"
    
    if not prompt_file.exists():
        raise HTTPException(status_code=404, detail=f"Prompt {category}/{name} not found")
    
    content = prompt_file.read_text()
    
    # Optionally append A2A skills if they exist
    a2a_skills_file = prompts_path / "common" / "a2a_skills.md"
    if a2a_skills_file.exists() and category != "common":
        a2a_content = a2a_skills_file.read_text()
        content = content + "\n\n---\n\n" + a2a_content
    
    # Apply runtime placeholder substitution
    content = substitute_placeholders(content)
    
    return {
        "role": "system",
        "content": content
    }


if __name__ == "__main__":
    print(f"Starting Prompt Registry on port {PROMPT_PORT}")
    print(f"Prompts directory: {PROMPTS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=PROMPT_PORT)

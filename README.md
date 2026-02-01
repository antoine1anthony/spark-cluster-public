# Spark-Cluster-001

A containerized, multi-node AI infrastructure using a **"Brain & Brawn"** architecture for distributed LLM inference, specialized AI services, and AI video generation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Node 1 "Brain"                                                   │
│  • Primary LLM (vLLM) ─────────────────────────────── Port 8000 │
│  • Prompt Registry ────────────────────────────────── Port 8010 │
│  • MCP Server ─────────────────────────────────────── Port 5050 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Node 2 "Utility" (Brawn)                                         │
│  • Auteur Video Worker (T2V/I2V) ──────────────────── Port 8000 │
│  • Math/Logic Prover (vLLM) ───────────────────────── Port 8005 │
│  • Whisper Speech-to-Text ─────────────────────────── Port 8007 │
│  • Weather Proxy ──────────────────────────────────── Port 8008 │
│  • A2A Gateway ────────────────────────────────────── Port 9000 │
└─────────────────────────────────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for detailed diagrams and data flow.

## Features

| Component | Description |
|-----------|-------------|
| **Brain Node** | Primary LLM with vLLM optimizations (FP4/FP8, chunked prefill, prefix caching) |
| **Utility Node** | Heavy compute: video generation, math proofs, transcription |
| **Auteur System** | AI Video Generation (HunyuanVideo T2V, SVD I2V) with ffprobe validation |
| **MCP Server** | Model Context Protocol for tool integration |
| **A2A Gateway** | Agent-to-Agent protocol for inter-service communication |
| **Prompt Registry** | Dynamic prompt management with `{{MODEL_NAME}}` substitution |
| **Prompt Browser** | Streamlit UI for browsing and copying system prompts |

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# Edit .env with your model IDs and settings
```

**Required variables:**
- `BRAIN_MODEL_ID` – HuggingFace model ID for the primary LLM
- `PROVER_MODEL_ID` – HuggingFace model ID for the prover

### 2. Start Brain Node

```bash
cd deploy/spark3_brain
docker compose up -d
```

### 3. Start Utility Belt Node

```bash
cd deploy/spark4_utility
docker compose up -d
```

### 4. Verify

```bash
# Brain Node
curl http://localhost:8000/v1/models   # Brain LLM

# Utility Node
curl http://localhost:8000/health      # Auteur Worker (video generation)
curl http://localhost:8005/v1/models   # Prover
curl http://localhost:8007/health      # Whisper
curl http://localhost:9000/health      # A2A Gateway
```

## Project Structure

```
spark-cluster-public/
├── auteur-system/              # AI Video Generation
│   ├── server.py               # FastAPI job queue
│   ├── generate.py             # Model inference
│   ├── Dockerfile.auteur
│   └── tests/
├── deploy/
│   ├── spark3_brain/           # Brain node
│   │   ├── docker-compose.yml
│   │   ├── prompt_loader.py    # Prompt registry service
│   │   └── entrypoint.sh
│   ├── spark4_utility/         # Utility node (Brawn)
│   │   ├── docker-compose.yml
│   │   ├── a2a_wrappers/       # A2A Gateway
│   │   └── weather_proxy.py
│   └── prompt_browser/         # Streamlit UI
│       └── app.py
├── mcp-server/                 # MCP tool server
│   ├── app/main.py
│   ├── Dockerfile
│   └── requirements.txt
├── prompts/                    # Your prompts go here
│   └── README.md
├── docs/
│   ├── architecture.md
│   └── sanitization.md
├── .env.example
├── .gitignore
└── LICENSE
```

## Prompts

Agent system prompts are **not included** in this repository. Add your own to the `prompts/` directory:

```
prompts/
├── coder/
│   └── full_stack_developer.md
├── research/
│   └── research_assistant.md
└── common/
    └── a2a_skills.md
```

Use `{{MODEL_NAME}}` in prompts for automatic model name substitution.

See [prompts/README.md](prompts/README.md) for format details.

## MCP Integration

Configure your MCP client:

```json
{
  "mcpServers": {
    "spark-cluster": {
      "url": "http://localhost:5050/mcp"
    }
  }
}
```

**Available tools:**
- `generate_video` – Text-to-video (HunyuanVideo)
- `generate_video_from_image` – Image-to-video (SVD)
- `weather_forecast`, `transcribe`, `prove`
- `kb_search`, `pdf_sync`
- `a2a_discover_agents`, `a2a_send_message`, `a2a_get_task`

## Configuration

All settings via environment variables. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAIN_MODEL_ID` | *(required)* | HuggingFace model ID |
| `PROVER_MODEL_ID` | *(required)* | Prover model ID |
| `AUTEUR_TOKEN` | `change-me-in-production` | Video worker auth token |
| `AUTEUR_URL` | `http://localhost:8000` | Video worker endpoint |
| `GPU_MEM_UTIL` | `0.85` | GPU memory utilization (0-1) |
| `MAX_MODEL_LEN` | `131072` | Maximum context length |
| `VLLM_IMAGE` | `nvcr.io/nvidia/vllm:25.11-py3` | vLLM container image |

See `.env.example` for the complete list.

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with appropriate drivers
- NVIDIA Container Toolkit
- (Optional) NGC account for NVIDIA containers
- (Optional) HuggingFace account for gated models

## Documentation

- [Architecture](docs/architecture.md) – Diagrams, ports, data flow
- [Sanitization](docs/sanitization.md) – Guidelines for contributors

## License

[MIT](LICENSE)

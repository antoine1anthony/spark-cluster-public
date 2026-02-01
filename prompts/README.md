# Agent Prompts

This directory contains agent system prompts for the cluster.

## Directory Structure

Organize prompts by category:

```
prompts/
├── coder/
│   ├── full_stack_developer.md
│   └── software_architect.md
├── research/
│   └── research_assistant.md
├── science/
│   └── physics_calculus.md
├── video/
│   └── director.md        # Video generation prompts for Auteur
└── common/
    └── a2a_skills.md      # Shared A2A capability definitions
```

## Prompt Format

Each prompt should be a Markdown file with:

```markdown
# Agent Title

## Role
Brief description of what this agent does.

## Capabilities
- Capability 1
- Capability 2

## Instructions
Detailed instructions for the agent...

## Examples
Optional examples of expected behavior.
```

## Placeholder Support

Prompts support runtime placeholder substitution:

| Placeholder | Replaced With |
|-------------|---------------|
| `{{MODEL_NAME}}` | Active model name from vLLM |

Example usage in a prompt:

```markdown
You are an AI assistant powered by {{MODEL_NAME}}.
```

## Adding Prompts

1. Create a new `.md` file in the appropriate category folder
2. Follow the format above
3. The Prompt Registry will automatically discover it

## Private Prompts

This directory is intentionally empty in the public repository. Add your own prompts locally:

```bash
# Example: Create a coding assistant prompt
mkdir -p prompts/coder
cat > prompts/coder/python_expert.md << 'EOF'
# Python Expert

## Role
You are an expert Python developer powered by {{MODEL_NAME}}.

## Capabilities
- Write clean, idiomatic Python code
- Debug and optimize existing code
- Explain complex concepts clearly

## Instructions
When helping with Python:
1. Follow PEP 8 style guidelines
2. Use type hints where appropriate
3. Include docstrings for functions
4. Suggest tests when relevant
EOF
```

## Accessing Prompts

### Via Prompt Registry API

```bash
# List all prompts
curl http://localhost:8010/prompts

# Get a specific prompt
curl http://localhost:8010/prompts/coder/python_expert

# Get as OpenAI-compatible system message
curl http://localhost:8010/prompts/system/coder/python_expert
```

### Via Prompt Browser

Run the Streamlit Prompt Browser for a visual interface:

```bash
cd deploy/prompt_browser
pip install streamlit requests
streamlit run app.py
```

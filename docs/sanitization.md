# Sanitization Guidelines

This document describes the sanitization rules applied to make this repository safe for public sharing.

## What Was Sanitized

### 1. IP Addresses and Hostnames

All internal IP addresses and hostnames have been replaced with environment variables:

| Original Pattern | Replacement |
|-----------------|-------------|
| `192.168.x.x` | `${SPARK3_HOST}` / `${SPARK4_HOST}` |
| Internal hostnames | `localhost` or env vars |

### 2. File Paths

Absolute paths have been replaced with relative paths or environment variables:

| Original Pattern | Replacement |
|-----------------|-------------|
| `/home/username/...` | `${CLUSTER_ROOT}` or `./` |
| `/mnt/hf_cache` | `${HF_CACHE_DIR}` |

### 3. Model IDs and Images

Specific model identifiers have been replaced with placeholders:

| Original | Replacement |
|----------|-------------|
| `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4` | `${BRAIN_MODEL_ID}` |
| Specific prover models | `${PROVER_MODEL_ID}` |

### 4. Secrets and Tokens

Authentication tokens are replaced with safe placeholders:

| Original Pattern | Replacement |
|-----------------|-------------|
| Production API tokens | `change-me-in-production` |
| `AUTEUR_TOKEN` | Environment variable with placeholder default |
| Internal network IPs | `localhost` defaults |

### 5. Removed Content

The following was excluded from the public repository:

- Private agent prompts (replaced with stub documentation)
- Debug logs and cursor configuration
- HuggingFace cache contents
- Any credentials or API keys
- Generated video outputs

## Verification Checklist

Before publishing, verify these patterns are not present:

```bash
# Check for private IPs
grep -rn "192\.168\." --include="*.yml" --include="*.py" --include="*.sh"

# Check for usernames in paths
grep -rn "/home/[a-z]" --include="*.yml" --include="*.py" --include="*.sh"

# Check for hardcoded model IDs (adjust pattern as needed)
grep -rn "nvidia/" --include="*.yml" --include="*.py"

# Check for tokens/keys
grep -rni "token\|api_key\|password\|secret" --include="*.yml" --include="*.py"
```

## Adding New Files

When contributing new files, ensure:

1. **No hardcoded IPs**: Use environment variables
2. **No absolute paths**: Use relative paths or env vars
3. **No credentials**: Use env vars for any secrets
4. **No private model IDs**: Document requirements, use placeholders

## Environment Variable Reference

All configurable values should be in `.env.example`:

```bash
# Required variables
BRAIN_MODEL_ID=       # HuggingFace model ID
PROVER_MODEL_ID=      # Prover model ID

# Service URLs (required for multi-node)
SPARK3_HOST=localhost
SPARK4_HOST=localhost
AUTEUR_URL=http://localhost:8000

# Secrets (MUST be changed in production)
AUTEUR_TOKEN=change-me-in-production

# Optional with defaults
GPU_MEM_UTIL=0.85
AUTEUR_OUTPUT_DIR=./auteur-output
```

## Secret Scanning Tools

Recommended tools for pre-commit verification:

- [TruffleHog](https://github.com/trufflesecurity/trufflehog)
- [git-secrets](https://github.com/awslabs/git-secrets)
- [detect-secrets](https://github.com/Yelp/detect-secrets)

Example pre-commit hook:

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Run trufflehog on staged files
git diff --cached --name-only | xargs trufflehog filesystem

# Check for IP addresses
if git diff --cached | grep -E "192\.168\.[0-9]+\.[0-9]+"; then
    echo "ERROR: Found private IP address in staged changes"
    exit 1
fi
```

# MCP Server Dockerfile
# Provides Model Context Protocol access to cluster tools

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Default port
EXPOSE 5050

# Start MCP server
CMD ["python", "-m", "app.main"]

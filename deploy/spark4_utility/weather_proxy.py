#!/usr/bin/env python3
"""Weather Simulation Proxy Service - Forwards requests to weather backend."""

from fastapi import FastAPI
import httpx
import uvicorn
import os

app = FastAPI(title="Weather Sim Proxy")

# Configure via environment variable
WEATHER_TOOL_URL = os.environ.get("WEATHER_TOOL_URL", "http://localhost:6000")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast/{region}")
async def forecast(region: str):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{WEATHER_TOOL_URL}/forecast/{region}")
        return resp.json()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)

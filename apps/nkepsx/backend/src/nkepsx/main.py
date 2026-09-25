#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "pydantic-settings",
# ]
# ///

"""
Main application entrypoint for the FastAPI orchestrator service.
This module initializes the FastAPI application instance, configures 
cross-origin resource sharing (CORS) middleware, 
and registers core system and routing endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from nkepsx.config import settings


# Initialize the core FastAPI application using metadata from centralized settings
app = FastAPI(title=settings.api_title, version=settings.api_version)

# Configure CORS middleware for local frontend integration and trusted domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": settings.api_version}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
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


from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from nkepsx.config import settings
from nkepsx.core.database import connect_to_mongo, close_mongo_connection, get_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB
    await connect_to_mongo()
    yield
    # Shutdown: Close connection
    await close_mongo_connection()

# Initialize the core FastAPI application using metadata from centralized settings
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan
)
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

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "nkepsx-backend"}

# Sample endpoint fetching data from your MongoDB instance
@app.get("/api/datasets")
async def get_datasets(client: AsyncIOMotorClient = Depends(get_database)):
    # Connects to your 'nkepsx' database and lists collections or documents
    db = client["nkepsx"]
    collections = await db.list_collection_names()
    
    # Example: fetch documents from a collection if it exists, or return metadata
    return {
        "collections": collections,
        "status": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
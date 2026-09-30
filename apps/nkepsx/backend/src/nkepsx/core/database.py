#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "motor",
#     "pydantic-settings",
#     "tritonclient[grpc]",
#     "numpy",
# ]
# ///

"""
Database connection and lifecycle manager for MongoDB using Motor.
"""

import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from nkepsx.config import settings

logger = logging.getLogger("uvicorn")

class Database:
    client: AsyncIOMotorClient | None = None

db = Database()

async def get_database() -> AsyncIOMotorClient:
    """Dependency helper to get the MongoDB client instance."""
    if db.client is None:
        raise RuntimeError("Database client is not initialized.")
    return db.client

async def connect_to_mongo(retries: int = 5, delay: int = 2) -> None:
    """Connect to MongoDB with retry logic on startup."""
    
    mongo_uri = settings.database_url
    db.client = AsyncIOMotorClient(
        mongo_uri,
        serverSelectionTimeoutMS=5000
    )
    
    for attempt in range(1, retries + 1):
        try:
            # The ping command is cheap and authenticates the connection
            await db.client.admin.command("ping")
            logger.info(f"Successfully connected to MongoDB at {mongo_uri}")
            return
        except ConnectionFailure as e:
            logger.warning(f"Attempt {attempt}/{retries}: MongoDB not ready yet ({e}). Retrying in {delay}s...")
            if attempt == retries:
                logger.error("Max retries reached. Could not connect to MongoDB.")
                raise e
            await asyncio.sleep(delay)

async def close_mongo_connection() -> None:
    """Close MongoDB connection on application shutdown."""
    if db.client:
        db.client.close()
        logger.info("Closed MongoDB connection.")
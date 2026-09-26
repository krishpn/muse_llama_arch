#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pymongo",
#     "motor",
#     "pydantic-settings",
# ]
# ///

"""
Standalone script to verify MongoDB connection and fetch a sample
of records from the insTrader.CRSP_mnthly_stk collection.
"""

import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient

async def main():
    mongo_uri = os.getenv("NKEPSX_DATABASE_URL", "mongodb://localhost:27017")
    print(f"Connecting to MongoDB at {mongo_uri}...")
    
    client = AsyncIOMotorClient(mongo_uri, serverSelectionTimeoutMS=5000)
    
    try:
        await client.admin.command("ping")
        print("Successfully connected to MongoDB!")
        
        db = client["insTrader"]
        collection = db["CRSP_mnthly_stk"]
        
        count = await collection.count_documents({})
        print(f"Total documents in insTrader.CRSP_mnthly_stk: {count}")
        
        cursor = collection.find().limit(30)
        records = await cursor.to_list(length=30)
        
        print(f"\nSuccessfully fetched {len(records)} sample records:")
        for idx, rec in enumerate(records[:5], 1):
            print(f"  [{idx}] ID: {rec.get('_id')} | Keys: {list(rec.keys())[:5]}...")
            
    except Exception as e:
        print(f"Error connecting or querying database: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(main())

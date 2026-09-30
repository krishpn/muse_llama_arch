#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pymongo",
#     "motor",
#     "asyncio",
#     "pydantic-settings",
# ]
# ///

"""
Standalone script to verify MongoDB connection and fetch a sample
of records from the insTrader.CRSP_mnthly_stk collection.
"""


import os, sys, logging, asyncio
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient




# Add src to python path so we can import config cleanly
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from nkepsx.config import settings

async def main():
    mongo_uri = settings.database_url
    db_name = settings.MONGO_DB_NAME

    print(f"Connecting to MongoDB at {mongo_uri}...")
    
    client = AsyncIOMotorClient(mongo_uri, serverSelectionTimeoutMS=5000)
    
    try:
        await client.admin.command("ping")
        print("MongoDB connection successful via NodePort!")
        
        db = client[db_name]
        target_collection = "CRSP_mnthly_stk"
        collection = db[target_collection]
        
        count = await collection.count_documents({})
        print(f"Collection '{target_collection}' in database '{db_name}' contains exactly {count:,} documents.")
        
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

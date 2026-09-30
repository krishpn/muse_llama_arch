from fastapi import APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
import os
from nkepsx.config import settings


router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/crsp-summary")
async def get_crsp_summary():
    try:
        # Connect or reference your client (adjust connection uri/db name as needed)
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:30017")
        client = AsyncIOMotorClient(mongo_uri)
        db = client["insTrader"]
        
        # Get collection document count
        count = await db["CRSP_mnthly_stk"].count_documents({})
        
        # Fetch a few sample records for previewing in the UI
        cursor = db["CRSP_mnthly_stk"].find({}, {"_id": 0}).limit(10)
        samples = await cursor.to_list(length=10)

        return {
            "status": "success",
            "database": "insTrader",
            "collection": "CRSP_mnthly_stk",
            "total_count": count,
            "sample_records": samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
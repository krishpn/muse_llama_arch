from datetime import datetime, timezone
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from nkepsx.config import settings
from nkepsx.core.database import get_database
from nkepsx.services.triton_service import TritonService
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["Orchestrator"])


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="User prompt for model inference")
    model_name: str = Field(
        default="llama_model", description="Target Triton model name"
    )


@router.post("/generate")
async def generate_response(
    payload: PromptRequest, db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Receive user prompt, log audit trail, invoke Triton backend with mTLS/gRPC, and stream results."""
    audit_record = {
        "prompt": payload.prompt,
        "model_name": payload.model_name,
        "timestamp": datetime.now(timezone.utc),
        "status": "initiated",
    }

    # 1. Log incoming request to MongoDB audit collection
    insert_result = await db.audit_logs.insert_one(audit_record)
    log_id = str(insert_result.inserted_id)

    try:
        # Mocking token input array conversion for LLM input tensors
        # (In production, replace with your actual tokenizer logic)
        input_tensor = np.array([[101, 2054, 2003, 102]], dtype=np.int32)

        # 2. Initialize Triton Service using centralized settings and optional mTLS certs
        triton_service = TritonService(
            grpc_url=settings.triton_grpc_url,
            verify_ssl=settings.verify_ssl,
            ca_cert=settings.ca_cert_path,
        )

        # 3. Stream inference generator
        async def response_generator():
            try:
                async for output_chunk in triton_service.infer_stream(
                    payload.model_name, input_tensor
                ):
                    yield output_chunk.tobytes()

                # Update audit log to completed upon successful stream finish
                await db.audit_logs.update_one(
                    {"_id": insert_result.inserted_id},
                    {"$set": {"status": "completed"}},
                )
            except Exception as stream_error:
                await db.audit_logs.update_one(
                    {"_id": insert_result.inserted_id},
                    {
                        "$set": {
                            "status": "failed",
                            "error": str(stream_error),
                        }
                    },
                )
                raise stream_error

        return StreamingResponse(
            response_generator(), media_type="application/octet-stream"
        )

    except Exception as e:
        await db.audit_logs.update_one(
            {"_id": insert_result.inserted_id},
            {"$set": {"status": "failed", "error": str(e)}},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failed: {str(e)}",
        )
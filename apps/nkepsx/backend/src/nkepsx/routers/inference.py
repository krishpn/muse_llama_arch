# apps/nkepsx/backend/src/nkepsx/services/inference.py
from nkepsx.core.clients import create_secure_http_client
from nkepsx.config import settings

async def call_vllm_model(prompt: str):
    async with create_secure_http_client() as client:
        response = await client.post(
            f"{settings.vllm_endpoint}/v1/completions",
            json={"prompt": prompt, "max_tokens": 100}
        )
        return response.json()
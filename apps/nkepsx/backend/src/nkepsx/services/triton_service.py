#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "tritonclient[grpc]",
#     "numpy",
#     "pydantic-settings",
# ]
# ///

"""
Asynchronous Triton Inference Server gRPC client with mTLS support.
"""

import ssl
from typing import AsyncGenerator
import numpy as np
import tritonclient.grpc.aio as grpcclient


class TritonService:
    def __init__(self, grpc_url: str, verify_ssl: bool = True, ca_cert: str | None = None):
        self.grpc_url = grpc_url.replace("grpcs://", "").replace("grpc://", "")
        self.verify_ssl = verify_ssl
        self.ca_cert = ca_cert
        self.client: grpcclient.InferenceServerClient | None = None

    async def connect(self):
        """Establish asynchronous gRPC connection to Triton."""
        ssl_options = None
        if self.verify_ssl and self.ca_cert:
            ssl_options = grpcclient.SslOptions(
                root_certificates=open(self.ca_cert, "rb").read()
            )
        
        self.client = grpcclient.InferenceServerClient(
            url=self.grpc_url,
            ssl_options=ssl_options,
            verbose=False
        )

    async def is_ready(self) -> bool:
        """Check if Triton server is live and ready."""
        if not self.client:
            await self.connect()
        return await self.client.is_server_ready()

    async def infer_stream(self, model_name: str, input_data: np.ndarray) -> AsyncGenerator[np.ndarray, None]:
        """Perform streaming inference against a deployed Triton model."""
        if not self.client:
            await self.connect()

        inputs = [grpcclient.InferInput("input_ids", input_data.shape, "INT32")]
        inputs[0].set_data_from_numpy(input_data)

        outputs = [grpcclient.InferRequestedOutput("output_ids")]

        response_stream = await self.client.stream_infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        async for result, error in response_stream:
            if error:
                raise RuntimeError(f"Triton inference error: {error}")
            yield result.as_numpy("output_ids")
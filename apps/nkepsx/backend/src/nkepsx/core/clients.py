#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx",
#     "pydantic-settings",
# ]
# ///

"""
Secure HTTP client factory module.

This module provides utility functions to instantiate asynchronous HTTP clients
pre-configured with enterprise PKI trust stores, custom CA bundles, and mTLS credentials.
"""

import httpx
from nkepsx.config import settings


def create_secure_http_client() -> httpx.AsyncClient:
    """
    Create an asynchronous HTTP client configured with enterprise PKI 
    custom CAs and mTLS credentials from application settings.

    Returns:
        httpx.AsyncClient: A configured async client instance ready for secure communication.
    """
    # 1. Determine verification target (custom enterprise CA bundle or standard SSL flag)
    verify_target: str | bool = (
        settings.ca_cert_path if settings.ca_cert_path else settings.verify_ssl
    )

    # 2. Package client certificates and private keys for mTLS if both are supplied
    cert_tuple: tuple[str, str] | None = (
        (settings.client_cert_path, settings.client_key_path)
        if settings.client_cert_path and settings.client_key_path
        else None
    )

    return httpx.AsyncClient(
        verify=verify_target,
        cert=cert_tuple,
        timeout=settings.inference_timeout_seconds,
    )
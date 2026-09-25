#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pydantic-settings",
# ]
# ///

"""
Centralized configuration and logging module for the FastAPI orchestrator.

This module uses Pydantic v2 BaseSettings to load and validate infrastructure 
endpoints, encryption standards, PKI certificate paths, and compliance 
flags from environment variables and `.env` files.
"""

import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings schema bound to environment variables."""

    # --- App Metadata & Environment ---
    api_title: str = "nkepsx API"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # --- AI Serving & Model Storage (Encrypted in flight & at rest) ---
    triton_grpc_url: str = "grpcs://localhost:8001"
    vllm_endpoint: str = "https://localhost:8000"
    model_repository_path: str = "/models"
    inference_timeout_seconds: float = 60.0

    # --- PKI, Certificates, & mTLS (Encryption in Flight) ---
    verify_ssl: bool = True
    ca_cert_path: str | None = None
    client_cert_path: str | None = None
    client_key_path: str | None = None

    # --- Encryption at Rest & Key Management (KMS) ---
    kms_key_arn: str | None = None  # AWS KMS / 
    encryption_algorithm: str = "AES-256-GCM"  # Approved cipher suite standard

    # --- Object Storage (S3 / MinIO - Encrypted at Rest via KMS) ---
    s3_endpoint_url: str | None = None
    s3_bucket_name: str = "nkepsx-artifacts"
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    s3_use_kms_encryption: bool = True  # Forces SSE-KMS on artifact uploads

    # --- Relational Database (RDS / PostgreSQL - Encrypted at Rest) ---
    #database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/nkepsx"
    database_url: str = "mongodb://nkepsx-mongodb-svc:27017/nkepsx"
    database_ssl_mode: str = "verify-full"  # Forces SSL verification for DB connection

    # --- Vector Store (Qdrant / Milvus - Encrypted Storage) ---
    vector_store_provider: str = "qdrant"
    vector_store_url: str = "https://localhost:6333"
    vector_store_api_key: str | None = None

    # --- Caching & Sessions (Redis - TLS enabled) ---
    redis_url: str = "rediss://localhost:6379/0"  # TLS-secured Redis

    # --- Observability & Distributed Tracing (OpenTelemetry) ---
    enable_telemetry: bool = True
    otel_exporter_otlp_endpoint: str | None = None  # e.g., http://otel-collector.monitoring.svc.cluster.local:4317

    # --- Audit & Security Compliance ---
    enable_structured_logging: bool = True  # Forces JSON logs for SIEM / Splunk ingestion
    audit_log_retention_days: int = 90

    model_config = SettingsConfigDict(
        env_prefix="NKEPSX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def api_version(self) -> str:
        """Dynamically fetch the package version or fallback to default."""
        try:
            return version("nkepsx")
        except PackageNotFoundError:
            return "0.1.0"


# Instantiate global settings singleton
settings = Settings()

# --- Centralized Global Logging Configuration ---
log_level_value = getattr(logging, settings.log_level.upper(), logging.INFO)

if settings.enable_structured_logging:
    log_format = '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
else:
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logging.basicConfig(
    stream=sys.stdout,
    level=log_level_value,
    format=log_format,
    force=True,
)
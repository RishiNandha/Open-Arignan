"""Ingestion package."""

from arignan.ingestion.discovery import discover_sources, is_web_url
from arignan.ingestion.log import IngestionLog
from arignan.ingestion.parsers import DocumentParser, FetchedUrl, HttpUrlFetcher, UrlFetcher
from arignan.ingestion.service import IngestionBatch, IngestionService, generate_load_id

__all__ = [
    "DocumentParser",
    "FetchedUrl",
    "HttpUrlFetcher",
    "IngestionBatch",
    "IngestionLog",
    "IngestionService",
    "UrlFetcher",
    "discover_sources",
    "generate_load_id",
    "is_web_url",
]

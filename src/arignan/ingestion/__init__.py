"""Ingestion package."""

from arignan.ingestion.discovery import discover_sources, is_web_url
from arignan.ingestion.log import IngestionLog
from arignan.ingestion.parsers import DocumentParser, FetchedUrl, HttpUrlFetcher, PdfOcrRequired, UrlFetcher
from arignan.ingestion.service import IngestionBatch, IngestionFailure, IngestionService, generate_load_id

__all__ = [
    "DocumentParser",
    "FetchedUrl",
    "HttpUrlFetcher",
    "IngestionBatch",
    "IngestionFailure",
    "IngestionLog",
    "IngestionService",
    "PdfOcrRequired",
    "UrlFetcher",
    "discover_sources",
    "generate_load_id",
    "is_web_url",
]

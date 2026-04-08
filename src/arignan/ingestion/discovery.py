from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from arignan.models import SourceDocument, SourceType

MARKDOWN_EXTENSIONS = {".md", ".markdown"}
PDF_EXTENSIONS = {".pdf"}


def is_web_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def discover_sources(input_ref: str | Path) -> list[SourceDocument]:
    raw_value = str(input_ref)
    if is_web_url(raw_value):
        return [SourceDocument(source_type=SourceType.URL, source_uri=raw_value)]

    path = Path(input_ref).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"input does not exist: {input_ref}")

    if path.is_dir():
        return _discover_from_directory(path)

    return [_source_from_file(path)]


def _discover_from_directory(directory: Path) -> list[SourceDocument]:
    discovered: list[SourceDocument] = []
    for candidate in sorted(directory.rglob("*")):
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix in MARKDOWN_EXTENSIONS or suffix in PDF_EXTENSIONS:
            discovered.append(_source_from_file(candidate))

    if not discovered:
        raise ValueError(f"no supported markdown or pdf files found in {directory}")
    return discovered


def _source_from_file(path: Path) -> SourceDocument:
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_EXTENSIONS:
        return SourceDocument(
            source_type=SourceType.MARKDOWN,
            source_uri=str(path),
            local_path=path,
            title=None,
        )
    if suffix in PDF_EXTENSIONS:
        return SourceDocument(
            source_type=SourceType.PDF,
            source_uri=str(path),
            local_path=path,
            title=path.stem,
        )
    raise ValueError(f"unsupported input type: {path}")

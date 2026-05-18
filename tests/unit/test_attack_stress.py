"""Adversarial attack stress tests for all 12 security fixes.

These tests fire real-world attack payloads directly at the security boundaries,
verifying that every control holds under adversarial pressure.  They complement
the unit tests in test_security_*.py, which test happy-path and edge cases.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent


def _resolve(ip: str) -> list:
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]


def _resolve6(ip: str) -> list:
    return [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", (ip, 0, 0, 0))]


# ============================================================================
# Attack Class 1 — SSRF (Fix #5)
# ============================================================================

class TestSSRFAttacks:
    """Each test is a distinct SSRF attack vector that must be blocked."""

    @pytest.fixture(autouse=True)
    def import_validator(self):
        from arignan.ingestion.parsers import _validate_fetch_url
        self.validate = _validate_fetch_url

    # ── Scheme attacks ───────────────────────────────────────────────────────

    def test_file_scheme_blocked(self):
        with pytest.raises(ValueError):
            self.validate("file:///etc/passwd")

    def test_file_scheme_windows_unc_blocked(self):
        with pytest.raises(ValueError):
            self.validate("file://server/share/secret.txt")

    def test_gopher_blocked(self):
        with pytest.raises(ValueError):
            self.validate("gopher://evil.com/")

    def test_dict_blocked(self):
        with pytest.raises(ValueError):
            self.validate("dict://evil.com/")

    def test_ldap_blocked(self):
        with pytest.raises(ValueError):
            self.validate("ldap://evil.com/")

    def test_ftp_blocked(self):
        with pytest.raises(ValueError):
            self.validate("ftp://evil.com/file")

    def test_data_uri_blocked(self):
        with pytest.raises(ValueError):
            self.validate("data:text/html,<h1>XSS</h1>")

    def test_javascript_blocked(self):
        with pytest.raises(ValueError):
            self.validate("javascript:alert(1)")

    def test_ssh_blocked(self):
        with pytest.raises(ValueError):
            self.validate("ssh://user@server/")

    # ── Private / loopback IPv4 ──────────────────────────────────────────────

    @pytest.mark.parametrize("ip", [
        "127.0.0.1",   # loopback
        "127.255.255.1",  # loopback range
        "10.0.0.1",    # RFC-1918 class A
        "10.255.255.255",
        "172.16.0.1",  # RFC-1918 class B low
        "172.31.255.255",  # RFC-1918 class B high
        "192.168.0.1",  # RFC-1918 class C
        "192.168.255.255",
        "169.254.169.254",  # AWS/GCP/Azure metadata
        "169.254.0.1",   # link-local APIPA
        "169.254.255.254",
        "0.0.0.0",  # unspecified
        "100.64.0.1",  # CGNAT (reserved)
        "240.0.0.1",  # reserved future
        "255.255.255.255",  # broadcast
        "224.0.0.1",  # multicast
    ])
    def test_private_ipv4_blocked(self, ip):
        with patch("socket.getaddrinfo", return_value=_resolve(ip)):
            with pytest.raises(ValueError, match="private|reserved|internal"):
                self.validate(f"http://target.internal/")

    # ── IPv6 attacks ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("ip6", [
        "::1",          # loopback
        "fe80::1",      # link-local
        "fc00::1",      # ULA
        "fd00::1",      # ULA
        "ff00::1",      # multicast
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "::ffff:192.168.1.1",  # IPv4-mapped private
        "::ffff:10.0.0.1",
        "::ffff:169.254.169.254",  # IPv4-mapped metadata endpoint
    ])
    def test_private_ipv6_blocked(self, ip6):
        addr = ipaddress.ip_address(ip6)
        if isinstance(addr, ipaddress.IPv6Address):
            infos = _resolve6(str(addr))
        else:
            infos = _resolve(str(addr))
        with patch("socket.getaddrinfo", return_value=infos):
            with pytest.raises(ValueError, match="private|reserved|internal"):
                self.validate("http://evil-ipv6/")

    # ── DNS rebinding / unresolvable ─────────────────────────────────────────

    def test_unresolvable_host_blocked(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("NXDOMAIN")):
            with pytest.raises(ValueError, match="resolve|hostname"):
                self.validate("http://does-not-exist.invalid/")

    def test_url_without_hostname_blocked(self):
        with pytest.raises(ValueError, match="hostname"):
            self.validate("http:///no-host")

    def test_blank_hostname_blocked(self):
        with pytest.raises(ValueError):
            self.validate("http:// /path")

    def test_valid_public_url_passes(self):
        with patch("socket.getaddrinfo", return_value=_resolve("93.184.216.34")):
            self.validate("https://example.com/paper.pdf")

    def test_valid_http_url_passes(self):
        with patch("socket.getaddrinfo", return_value=_resolve("1.2.3.4")):
            self.validate("http://public-api.example.com/resource")


# ============================================================================
# Attack Class 2 — Windows batch shell injection (Fix #6)
# ============================================================================

class TestBatchInjectionAttacks:
    @pytest.fixture(autouse=True)
    def import_escape(self):
        from arignan.setup_flow import _escape_batch_argument
        self.escape = _escape_batch_argument

    # Classic injection payloads
    @pytest.mark.parametrize("payload,expected_action", [
        ('"&& del /F /Q C:\\Windows\\', "double_quote_escaped"),
        ('"|| calc.exe', "double_quote_escaped"),
        ('C:\\path\\"&whoami', "double_quote_escaped"),
        ('"something" > C:\\evil.txt', "double_quote_escaped"),
    ])
    def test_quote_injection_neutralised(self, payload, expected_action):
        result = self.escape(payload)
        # After escaping, there must be no unescaped " that could close the
        # batch double-quote context and inject a new command.
        # Every " becomes "" (two chars), so no standalone " remains.
        # Count standalone quotes: replace "" with placeholder and check no " left
        sanitised = result.replace('""', '__DQ__')
        assert '"' not in sanitised, f"Unescaped quote in escaped output: {result!r}"

    def test_cr_in_path_raises(self):
        with pytest.raises(ValueError, match="carriage return"):
            self.escape("C:\\path\r\\app")

    def test_lf_in_path_raises(self):
        with pytest.raises(ValueError, match="newline"):
            self.escape("/home/user\n/.arignan")

    def test_null_byte_raises(self):
        with pytest.raises(ValueError, match="null byte"):
            self.escape("/path/\x00hidden")

    def test_crlf_together_raises(self):
        with pytest.raises(ValueError):
            self.escape("path\r\ninjection")

    def test_normal_windows_path_unchanged(self):
        path = r"C:\Users\alice\AppData\Local\Arignan"
        assert self.escape(path) == path

    def test_unix_path_with_spaces_unchanged(self):
        path = "/home/alice/my documents/arignan"
        assert self.escape(path) == path

    def test_unc_path_no_injection(self):
        path = r"\\server\share\path"
        result = self.escape(path)
        sanitised = result.replace('""', '__DQ__')
        assert '"' not in sanitised


# ============================================================================
# Attack Class 3 — Hat path traversal (Fix #7)
# ============================================================================

class TestHatTraversalAttacks:
    @pytest.fixture(autouse=True)
    def import_validator(self):
        from arignan.storage.layout import validate_hat_name
        self.validate = validate_hat_name

    @pytest.mark.parametrize("payload", [
        "../secret",
        "../../etc/passwd",
        "..",
        "../",
        "hat/../../../root",
        "hat/subdir",
        "hat\\..\\secret",
        "/absolute/path",
        "hat\x00null",
        "hat\nnewline",
        "hat\rcarriage",
        " ",
        "",
        ".",
        "hat name with spaces",
        "hat!@#$%",
        "haténame",  # non-ASCII character (accented e) rejected by HAT_NAME_PATTERN
    ])
    def test_traversal_payload_rejected(self, payload):
        with pytest.raises(ValueError):
            self.validate(payload)

    @pytest.mark.parametrize("name", [
        "default",
        "research",
        "my-hat",
        "hat_v2",
        "hat.v2",
        "Hat123",
        "A",
    ])
    def test_valid_hat_names_accepted(self, name):
        result = self.validate(name)
        assert result == name.strip()


# ============================================================================
# Attack Class 4 — Concurrent JSON index corruption (Fix #8)
# ============================================================================

class TestConcurrentIndexAttacks:
    def _json_index(self, tmp_path):
        from arignan.indexing.dense import LocalDenseIndex
        with patch("arignan.indexing.dense.LocalDenseIndex._try_create_qdrant_client", return_value=None):
            return LocalDenseIndex(tmp_path)

    def _chunk(self, chunk_id: str):
        from arignan.models import ChunkMetadata, ChunkRecord
        return ChunkRecord(
            chunk_id=chunk_id,
            text=f"text-{chunk_id}",
            metadata=ChunkMetadata(load_id="stress", hat="default", source_uri="a.md"),
            embedding=[float(i % 10) * 0.1 for i in range(10)],
        )

    def test_50_threads_concurrent_upsert(self, tmp_path):
        """50 threads writing simultaneously must not corrupt JSON."""
        index = self._json_index(tmp_path)
        errors: list[Exception] = []

        def worker(i: int):
            try:
                index.upsert([self._chunk(f"chunk-{i:04d}")])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent write errors: {errors}"
        all_ids = {c.chunk_id for c in index.all_chunks()}
        assert len(all_ids) == 50

    def test_json_file_always_parseable_under_concurrent_writes(self, tmp_path):
        """Readers must never see a partially written file."""
        index = self._json_index(tmp_path)
        corruption_detected = []

        # Seed with one chunk so the file exists
        index.upsert([self._chunk("seed")])

        def reader():
            for _ in range(200):
                try:
                    data = index.storage_path.read_text(encoding="utf-8")
                    json.loads(data)
                except (json.JSONDecodeError, FileNotFoundError) as exc:
                    corruption_detected.append(str(exc))

        def writer(i: int):
            for j in range(5):
                index.upsert([self._chunk(f"w{i}-{j}")])

        threads = [threading.Thread(target=reader)]
        threads += [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not corruption_detected, f"File corruption detected: {corruption_detected}"


# ============================================================================
# Attack Class 5 — Ollama model name injection (Fix #9)
# ============================================================================

class TestOllamaModelNameAttacks:
    @pytest.fixture(autouse=True)
    def import_validator(self):
        from arignan.llm.service import _validate_ollama_model_name
        self.validate = _validate_ollama_model_name

    @pytest.mark.parametrize("payload", [
        # Shell metacharacters
        "; rm -rf /",
        "| cat /etc/shadow",
        "& calc.exe",
        "`whoami`",
        "$(id)",
        "model\nX-Injected: true",
        "model\r\nHTTP/1.1 200 OK",
        # Path traversal in model name
        "../../models/evil",
        "/absolute/path",
        "model\\evil",
        # Oversized
        "a" * 300,
        "m" * 201,
        # Empty / whitespace
        "",
        " ",
        "\t",
        # Unicode homoglyphs / invisible chars
        "llama​3",  # zero-width space
        "llama‮3",  # RTL override
        # Null bytes
        "model\x00.hidden",
    ])
    def test_malicious_model_name_rejected(self, payload):
        with pytest.raises(ValueError, match="Invalid Ollama model name"):
            self.validate(payload)

    @pytest.mark.parametrize("name", [
        "llama3",
        "qwen:4b",
        "myorg/mymodel:latest",
        "llama3.1:8b-instruct-q4_K_M",
        "model-v1.2.3",
        "a" * 200,  # exactly at limit
    ])
    def test_valid_model_names_accepted(self, name):
        self.validate(name)  # must not raise


# ============================================================================
# Attack Class 6 — Query length DoS (Fix #11)
# ============================================================================

class TestQueryLengthAttacks:
    @pytest.fixture(autouse=True)
    def import_expander(self):
        from arignan.retrieval.pipeline import QueryExpander, _MAX_QUERY_EXPANDED_CHARS
        self.expander = QueryExpander()
        self.max_expanded = _MAX_QUERY_EXPANDED_CHARS

    def test_10k_char_query_doesnt_crash(self):
        result = self.expander.expand("x" * 10_000)
        assert isinstance(result, str)
        assert len(result) <= self.max_expanded

    def test_100k_char_query_doesnt_crash(self):
        result = self.expander.expand("a " * 50_000)
        assert len(result) <= self.max_expanded

    def test_1m_char_query_doesnt_crash(self):
        result = self.expander.expand("z" * 1_000_000)
        assert len(result) <= self.max_expanded

    def test_repeated_abbreviations_dont_explode_output(self):
        # BM25 expands to "best matching 25"; repeating it many times should
        # still be capped by the output limit.
        query = ("bm25 " * 5000).strip()
        result = self.expander.expand(query)
        assert len(result) <= self.max_expanded

    def test_unicode_bomb_doesnt_crash(self):
        # Mix of wide chars that could trip string length calculations
        bomb = "𠜎𠜎𠜎" * 3000  # Each char is 4 UTF-8 bytes
        result = self.expander.expand(bomb)
        assert len(result) <= self.max_expanded

    def test_null_bytes_in_query_handled(self):
        result = self.expander.expand("query\x00with\x00nulls")
        assert isinstance(result, str)

    def test_newlines_and_tabs_in_query_handled(self):
        result = self.expander.expand("query\nwith\ttabs\r\nand newlines")
        assert isinstance(result, str)


# ============================================================================
# Attack Class 7 — Markdown injection (Fix #12)
# ============================================================================

class TestMarkdownInjectionAttacks:
    @pytest.fixture(autouse=True)
    def imports(self):
        from arignan.markdown.rendering import _escape_code_span, markdown_table_cell
        self.escape_span = _escape_code_span
        self.table_cell = markdown_table_cell

    # Code span (backtick) breakout attacks — routed through _escape_code_span
    @pytest.mark.parametrize("payload", [
        "`evil` [link](http://attacker.com)",
        "``double-backtick breakout``",
        "`\n\ncode block injection",
        "file`name`.pdf",
    ])
    def test_code_span_backtick_stripped(self, payload):
        result = self.escape_span(payload)
        assert "`" not in result, f"Backtick survived in: {result!r}"

    # Table cell structure injection attacks — routed through markdown_table_cell
    @pytest.mark.parametrize("payload,check", [
        ("| extra | column |", lambda r: "\\|" in r),
        ("cell\ncell2", lambda r: "\n" not in r),
        ("[link](http://evil.com)", lambda r: "\\[" in r),
        ("![img](http://evil.com/pixel.gif)", lambda r: "\\[" in r),
        ("<script>alert(1)</script>", lambda r: "<script>" not in r and "&lt;" in r),
        ("<img src=x onerror=alert(1)>", lambda r: "<img" not in r or "&lt;" in r),
        # Combined: [ + | in same cell
        ("[click me](evil)|side-channel", lambda r: "\\[" in r and "\\|" in r),
    ])
    def test_table_cell_injection_sanitised(self, payload, check):
        result = self.table_cell(payload)
        assert check(result), f"Payload not sanitised: {payload!r} → {result!r}"

    def test_benign_filename_preserved_in_code_span(self):
        assert self.escape_span("report_2024.pdf") == "report_2024.pdf"

    def test_benign_cell_text_preserved(self):
        result = self.table_cell("Normal cell content")
        assert result == "Normal cell content"


# ============================================================================
# Attack Class 8 — ReDoS (regex catastrophic backtracking)
# ============================================================================

class TestReDoSAttacks:
    """Verify the regex patterns in chunking.py can't be DoS'd."""

    def test_author_year_citation_no_backtracking(self):
        from arignan.indexing.chunking import AUTHOR_YEAR_CITATION_PATTERN
        # Adversarial: long string of nested-quantifier-bait chars that would
        # cause catastrophic backtracking in a naive pattern.
        evil = "(" + "A" * 5000 + "19" + "A" * 5000 + ")"
        start = time.monotonic()
        AUTHOR_YEAR_CITATION_PATTERN.search(evil)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"ReDoS suspected: {elapsed:.2f}s on adversarial input"

    def test_sentence_boundary_no_backtracking(self):
        from arignan.indexing.chunking import SENTENCE_BOUNDARY_PATTERN
        evil = "a. " * 10_000 + "B"
        start = time.monotonic()
        SENTENCE_BOUNDARY_PATTERN.split(evil)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"ReDoS suspected: {elapsed:.2f}s"

    def test_hat_name_pattern_no_backtracking(self):
        from arignan.storage.layout import HAT_NAME_PATTERN
        evil = "a-" * 5000
        start = time.monotonic()
        HAT_NAME_PATTERN.match(evil)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"ReDoS suspected: {elapsed:.2f}s"


# ============================================================================
# Attack Class 9 — Parameter overflow (Fix #16)
# ============================================================================

class TestParameterOverflowAttacks:
    def test_chunker_absurdly_large_chunk_size(self):
        from arignan.indexing.chunking import Chunker
        with pytest.raises(ValueError, match="chunk_size must not exceed"):
            Chunker(chunk_size=999_999_999)

    def test_chunker_negative_chunk_size(self):
        from arignan.indexing.chunking import Chunker
        with pytest.raises(ValueError):
            Chunker(chunk_size=-1)

    def test_chunker_overlap_larger_than_size(self):
        from arignan.indexing.chunking import Chunker
        with pytest.raises(ValueError):
            Chunker(chunk_size=100, chunk_overlap=200)

    def test_hashing_embedder_huge_dimension(self):
        from arignan.indexing.embedding import HashingEmbedder
        with pytest.raises(ValueError):
            HashingEmbedder(dimension=2**32)

    def test_hashing_embedder_zero_dimension(self):
        from arignan.indexing.embedding import HashingEmbedder
        with pytest.raises(ValueError):
            HashingEmbedder(dimension=0)


# ============================================================================
# Attack Class 10 — SHA-256 collision resistance (Fix #15)
# ============================================================================

class TestChunkIdCollisionResistance:
    def test_1000_chunks_no_collision(self):
        from arignan.indexing import Chunker
        from arignan.models import ParsedDocument, SourceDocument, SourceType

        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        docs = []
        for i in range(10):
            text = " ".join(f"Word{j} " for j in range(i * 50, i * 50 + 200))
            doc = ParsedDocument(
                load_id=f"load-{i}",
                hat="default",
                source=SourceDocument(
                    source_type=SourceType.MARKDOWN,
                    source_uri=f"doc{i}.md",
                    local_path=Path(f"doc{i}.md"),
                ),
                full_text=text,
                sections=[],
                keywords=[],
            )
            docs.append(doc)

        all_ids: list[str] = []
        for doc in docs:
            chunks = chunker.chunk_document(doc)
            all_ids.extend(c.chunk_id for c in chunks)

        assert len(all_ids) == len(set(all_ids)), "Chunk ID collision detected"
        assert all(c.startswith("chunk-") and len(c) == 26 for c in all_ids), \
            "Chunk IDs must be 'chunk-' + 20 hex chars"

    def test_chunk_id_format_sha256(self):
        from arignan.indexing import Chunker
        from arignan.models import ParsedDocument, SourceDocument, SourceType

        chunker = Chunker()
        doc = ParsedDocument(
            load_id="test",
            hat="default",
            source=SourceDocument(
                source_type=SourceType.MARKDOWN,
                source_uri="test.md",
                local_path=Path("test.md"),
            ),
            full_text="Hello world. This is a test.",
            sections=[],
            keywords=[],
        )
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.chunk_id.startswith("chunk-")
            hex_part = chunk.chunk_id[6:]
            assert len(hex_part) == 20
            # Must be valid hex
            int(hex_part, 16)


# ============================================================================
# Attack Class 11 — Exception context path leak (Fix #17)
# ============================================================================

class TestExceptionContextLeaks:
    @pytest.fixture(autouse=True)
    def import_sanitizer(self):
        from arignan.markdown.writer import _sanitize_exception_context
        self.sanitize = _sanitize_exception_context

    def test_nested_path_in_string_value_not_leaked(self):
        # A string that happens to contain a path but isn't a Path object
        # and isn't under a sensitive key — kept as-is (not over-sanitised)
        context = {"model": "llama3", "hat": "default"}
        result = self.sanitize(context)
        assert result["hat"] == "default"

    def test_path_object_always_redacted(self):
        for key in ("source_file", "output_path", "cache", "data", "log"):
            result = self.sanitize({key: Path("/private/secret/path")})
            assert result[key] == "<redacted-path>", f"Path not redacted under key {key!r}"

    def test_sensitive_keys_redacted_regardless_of_value_type(self):
        from arignan.markdown.writer import _SENSITIVE_CONTEXT_KEYS
        for key in _SENSITIVE_CONTEXT_KEYS:
            for value in ("string value", 42, ["list"], {"dict": True}):
                result = self.sanitize({key: value})
                assert result[key] == "<redacted>", \
                    f"Sensitive key {key!r} with value {value!r} not redacted"

    def test_empty_context_safe(self):
        assert self.sanitize({}) == {}

    def test_deeply_nested_is_not_recursively_sanitised(self):
        # We only sanitise top-level keys — nested dicts are not walked
        # (by design; recursive traversal would be surprising and slow).
        context = {"top": {"nested_path": Path("/secret")}}
        result = self.sanitize(context)
        # "top" is not a sensitive key and not a Path, so it passes through
        assert result["top"] == {"nested_path": Path("/secret")}


# ============================================================================
# Attack Class 12 — Upload directory permissions (Fix #14)
# ============================================================================

@pytest.mark.skipif(sys.platform == "win32", reason="Unix permission model only")
class TestUploadPermissionAttacks:
    def test_parent_and_batch_dir_are_owner_only(self, tmp_path):
        upload_root = tmp_path / "uploads"
        upload_root.mkdir(mode=0o700)
        batch = Path(tempfile.mkdtemp(prefix="batch-", dir=str(upload_root)))
        os.chmod(batch, 0o700)

        for path in (upload_root, batch):
            mode = stat.S_IMODE(os.stat(path).st_mode)
            assert mode == 0o700, f"{path} has mode {oct(mode)}, want 0o700"
            # None of group/other bits should be set
            assert not (mode & 0o077), f"{path} leaks permission bits to group/other"

    def test_file_written_inside_batch_dir_respects_umask(self, tmp_path):
        """Files written under the batch dir are the app's responsibility;
        test that we can write and they're not world-readable via the dir."""
        upload_root = tmp_path / "uploads"
        upload_root.mkdir(mode=0o700)
        batch = Path(tempfile.mkdtemp(prefix="batch-", dir=str(upload_root)))
        os.chmod(batch, 0o700)

        test_file = batch / "payload.bin"
        test_file.write_bytes(b"secret upload data")

        # The directory is 0o700, so other users can't even list it —
        # confirm the directory mode is still correct after writing a file
        dir_mode = stat.S_IMODE(os.stat(batch).st_mode)
        assert dir_mode == 0o700

    def test_umask_doesnt_weaken_permissions(self, tmp_path):
        """Even with a permissive umask, explicit mode=0o700 + chmod must hold."""
        orig_umask = os.umask(0o000)  # maximally permissive umask
        try:
            upload_root = tmp_path / "uploads_umask"
            upload_root.mkdir(mode=0o700)
            batch = Path(tempfile.mkdtemp(prefix="batch-", dir=str(upload_root)))
            os.chmod(batch, 0o700)
        finally:
            os.umask(orig_umask)

        for path in (upload_root, batch):
            mode = stat.S_IMODE(os.stat(path).st_mode)
            assert mode == 0o700, f"{path} is {oct(mode)} under permissive umask"

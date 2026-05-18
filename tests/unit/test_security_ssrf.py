"""Tests for SSRF protection in the URL fetcher (Fix #5)."""
from __future__ import annotations

import ipaddress
import socket
from unittest.mock import MagicMock, patch

import pytest

from arignan.ingestion.parsers import _validate_fetch_url


class TestValidateFetchUrl:
    def test_valid_https_url_passes(self) -> None:
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            _validate_fetch_url("https://example.com/page")

    def test_valid_http_url_passes(self) -> None:
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            _validate_fetch_url("http://example.com/page")

    def test_file_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            _validate_fetch_url("file:///etc/passwd")

    def test_ftp_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            _validate_fetch_url("ftp://example.com/file")

    def test_data_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            _validate_fetch_url("data:text/html,<h1>hi</h1>")

    def test_loopback_ipv4_rejected(self) -> None:
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]
            with pytest.raises(ValueError, match="private|reserved|internal"):
                _validate_fetch_url("http://localhost/")

    def test_loopback_127_x_x_x_rejected(self) -> None:
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.2", 0))]
            with pytest.raises(ValueError, match="private|reserved|internal"):
                _validate_fetch_url("http://loopback2/")

    def test_private_10_range_rejected(self) -> None:
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))]
            with pytest.raises(ValueError, match="private|reserved|internal"):
                _validate_fetch_url("http://internal.corp/")

    def test_private_172_range_rejected(self) -> None:
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("172.16.0.5", 0))]
            with pytest.raises(ValueError, match="private|reserved|internal"):
                _validate_fetch_url("http://priv/")

    def test_private_192_168_range_rejected(self) -> None:
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 0))]
            with pytest.raises(ValueError, match="private|reserved|internal"):
                _validate_fetch_url("http://router/")

    def test_cloud_metadata_endpoint_rejected(self) -> None:
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 0))]
            with pytest.raises(ValueError, match="private|reserved|internal"):
                _validate_fetch_url("http://169.254.169.254/latest/meta-data/")

    def test_link_local_rejected(self) -> None:
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.1.1", 0))]
            with pytest.raises(ValueError, match="private|reserved|internal"):
                _validate_fetch_url("http://link-local/")

    def test_ipv6_loopback_rejected(self) -> None:
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 0, 0, 0))]
            with pytest.raises(ValueError, match="private|reserved|internal"):
                _validate_fetch_url("http://[::1]/")

    def test_ipv6_link_local_rejected(self) -> None:
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("fe80::1", 0, 0, 0))]
            with pytest.raises(ValueError, match="private|reserved|internal"):
                _validate_fetch_url("http://[fe80::1]/")

    def test_dns_resolution_failure_raises(self) -> None:
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
            with pytest.raises(ValueError, match="resolve|hostname"):
                _validate_fetch_url("http://does-not-exist.invalid/")

    def test_missing_hostname_raises(self) -> None:
        with pytest.raises(ValueError, match="hostname"):
            _validate_fetch_url("http:///path")

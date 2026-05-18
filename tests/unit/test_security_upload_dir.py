"""Tests for upload directory permission hardening (Fix #14)."""
from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="Unix permission model only")
class TestUploadDirPermissions:
    def test_mkdir_mode_0o700_is_owner_only(self, tmp_path: Path) -> None:
        upload_root = tmp_path / "gui_uploads"
        upload_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        mode = stat.S_IMODE(os.stat(upload_root).st_mode)
        assert mode == 0o700, f"Expected mode 0o700, got {oct(mode)}"

    def test_mkdtemp_chmod_0o700_is_owner_only(self, tmp_path: Path) -> None:
        upload_root = tmp_path / "gui_uploads"
        upload_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        batch_dir = Path(tempfile.mkdtemp(prefix="batch-", dir=str(upload_root)))
        os.chmod(batch_dir, 0o700)
        mode = stat.S_IMODE(os.stat(batch_dir).st_mode)
        assert mode == 0o700, f"Expected mode 0o700, got {oct(mode)}"

    def test_upload_root_not_world_readable(self, tmp_path: Path) -> None:
        upload_root = tmp_path / "gui_uploads"
        upload_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        mode = stat.S_IMODE(os.stat(upload_root).st_mode)
        assert not (mode & stat.S_IROTH), "Upload root must not be world-readable"
        assert not (mode & stat.S_IWOTH), "Upload root must not be world-writable"
        assert not (mode & stat.S_IXOTH), "Upload root must not be world-executable"

    def test_batch_dir_not_group_readable(self, tmp_path: Path) -> None:
        upload_root = tmp_path / "gui_uploads"
        upload_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        batch_dir = Path(tempfile.mkdtemp(prefix="batch-", dir=str(upload_root)))
        os.chmod(batch_dir, 0o700)
        mode = stat.S_IMODE(os.stat(batch_dir).st_mode)
        assert not (mode & stat.S_IRGRP), "Batch dir must not be group-readable"
        assert not (mode & stat.S_IWGRP), "Batch dir must not be group-writable"

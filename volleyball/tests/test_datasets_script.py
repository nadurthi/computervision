"""datasets_to_s3.py against a local HTTP server and a fake gdown, with a local directory as destination."""
import functools
import http.server
import json
import sys
import threading
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import datasets_to_s3 as D  # noqa: E402


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    (root / "scout").mkdir(parents=True)
    (root / "scout" / "&a b.dvw").write_bytes(b"[3SCOUT]\n" * 1000)
    (root / "clip.mp4").write_bytes(bytes(range(256)) * 4096)        # 1 MB
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()


def test_http_stream_manifest_and_skip(tmp_path, server, monkeypatch):
    d = D.Dataset("toy", "http", "MIT", "toy", files=[(server + "clip.mp4", "video/clip.mp4"),
                                                     (server + D._q("scout/&a b.dvw"), "scout/&a b.dvw")],
                  commercial_ok=True)
    monkeypatch.setitem(D.REGISTRY, "toy", d)
    out = tmp_path / "bucket" / "datasets"
    rep = D.run(["toy"], str(out))
    assert rep["toy"] == {"files": 2, "bytes": 1048576 + 9000}
    assert (out / "toy/raw/video/clip.mp4").stat().st_size == 1048576
    m = json.loads((out / "toy/MANIFEST.json").read_text())
    assert m["licence"] == "MIT" and m["commercial_ok"] is True
    assert all(len(f["sha256"]) == 64 for f in m["files"])
    # second run: same sizes in the destination -> skipped (no hash recomputed)
    m2 = D.run(["toy"], str(out))
    m2f = json.loads((out / "toy/MANIFEST.json").read_text())["files"]
    assert m2["toy"]["files"] == 2 and all(f["sha256"] == "" for f in m2f)


def test_gdrive_extract_upload_and_cleanup(tmp_path, monkeypatch):
    calls = {}

    def fake_download(id, output, quiet=False):
        calls["id"] = id
        with zipfile.ZipFile(output, "w") as z:
            z.writestr("frames/0001.jpg", b"x" * 10)
            z.writestr("annotations.txt", b"a b c")

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=fake_download))
    d = D.Dataset("toydrive", "gdrive_file", "n/a", "toy", files=[("FILEID", "toy.zip")])
    monkeypatch.setitem(D.REGISTRY, "toydrive", d)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    out = tmp_path / "dest"
    rep = D.run(["toydrive"], str(out), extract=True, scratch=str(scratch))
    assert calls["id"] == "FILEID"
    assert (out / "toydrive/raw/toy/frames/0001.jpg").exists()
    assert rep["toydrive"]["files"] == 2
    assert list(scratch.iterdir()) == []                  # scratch cleaned up


def test_registry_is_consistent():
    for name, d in D.REGISTRY.items():
        assert d.name == name
        assert d.kind in {"http", "gdrive_folder", "gdrive_file", "soccernet", "roboflow", "manual_url"}
        if d.kind == "http":
            assert all(u.startswith("https://raw.githubusercontent.com/") for u, _ in d.files)
        if d.kind == "roboflow":
            assert {"workspace", "project"} <= set(d.extra)
    assert D.REGISTRY["sportsmot"].commercial_ok is False


def test_manual_url_requires_url(tmp_path):
    with pytest.raises(SystemExit):
        D.run(["sportsmot"], str(tmp_path))

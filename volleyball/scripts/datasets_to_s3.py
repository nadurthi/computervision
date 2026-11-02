"""Download public volleyball / jersey datasets straight into S3.

Run it on a SageMaker Studio space (fast AWS network, credentials from the execution role):

    python scripts/datasets_to_s3.py --list
    python scripts/datasets_to_s3.py --dest s3://vb-data-<account>/datasets ovdata pydatavolley_dvw
    python scripts/datasets_to_s3.py --dest s3://vb-data-<account>/datasets volleyball_ibrahim --extract
    python scripts/datasets_to_s3.py --dest s3://... sportsmot --url "<link from Codalab 'Get Data'>"
    ROBOFLOW_API_KEY=... python scripts/datasets_to_s3.py --dest s3://... roboflow_ball_daniel

What "directly to S3" means per source:
  http         streamed from the web into a multipart S3 upload, never written to disk
  gdrive_file  single Drive files: streamed straight into S3 with --direct (or when gdown is not
               installed); otherwise gdown writes to a scratch dir, then upload + cleanup
  gdrive_folder Drive folders need gdown: scratch dir, upload, cleanup (needs enough disk)
  soccernet    the SoccerNet downloader writes to scratch, then the same upload + cleanup
  roboflow     the Roboflow SDK writes to scratch, then the same upload + cleanup

Every dataset gets s3://.../<name>/raw/... plus <name>/MANIFEST.json with source, licence, date,
and each file's size and sha256. Files already in S3 with the same size are skipped (--force redoes).
--dest may also be a local directory (used by the tests).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

GH_OVDATA = "https://raw.githubusercontent.com/openvolley/ovdata/master/inst/extdata/"
GH_PYDV = "https://raw.githubusercontent.com/openvolley/py-datavolley/main/datavolley/data/"
# single Google Drive files can be fetched without gdown; confirm=t skips the large-file warning page
DRIVE_DIRECT = os.environ.get("VB_DRIVE_DIRECT_URL", "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t")


def _q(name: str) -> str:
    return urllib.parse.quote(name)


@dataclass
class Dataset:
    name: str
    kind: str                     # http | gdrive_folder | gdrive_file | soccernet | roboflow | manual_url
    licence: str
    note: str
    files: list[tuple[str, str]] = field(default_factory=list)   # (url or drive id, key under raw/)
    folder: str = ""              # gdrive folder id
    extra: dict = field(default_factory=dict)
    commercial_ok: bool | None = None   # None = not stated by the source; check before commercial use


REGISTRY: dict[str, Dataset] = {d.name: d for d in [
    Dataset("ovdata", "http", "MIT (openvolley/ovdata)",
            "A rally clip from Katowice vs Bedzin (PlusLiga 2018/19) with its DataVolley file, plus 4 full "
            "scouted matches. The only public pair of video + .dvw for the same rally: use it for the first "
            "event-level check of the pipeline.",
            files=[(GH_OVDATA + "video/2019_03_01-KATS-BEDS-clip.mp4", "video/2019_03_01-KATS-BEDS-clip.mp4"),
                   (GH_OVDATA + "images/2019_03_01-KATS-BEDS-frame.png", "images/2019_03_01-KATS-BEDS-frame.png")]
            + [(GH_OVDATA + "scout/" + _q(n), "scout/" + n) for n in
               ["&190301_kats_beds-clip.dvw", "&190301_kats_beds.dvw", "&DCup-7.dvw", "&stuttgart-schwerin-2018.dvw",
                "PM06.dvw", "mlafin_braslovce_nkbm.dvw"]],
            commercial_ok=True),
    Dataset("pydatavolley_dvw", "http", "MIT (openvolley/py-datavolley)",
            "The 7 expert-scouted matches bundled with py-datavolley; the skill rule is scored on these.",
            files=[(GH_PYDV + _q(n), n) for n in
                   ["&big102014-09-26 northweste-rutgers.dvw", "&ncaa2015-12-04 ohio state - robert morrison brup 4676.dvw",
                    "&ncaa2015-12-04 stanford - lmu tw 4673.dvw", "&ncaa2015-12-12 washington-nebraska 4705.dvw",
                    "PM06.dvw", "example_match.dvw", "testing_truncated.dvw"]],
            commercial_ok=True),
    Dataset("volleyball_ibrahim", "gdrive_folder",
            "Not stated for the data (the code is BSD-2-Clause); cite Ibrahim et al. CVPR 2016 and TPAMI",
            "Volleyball dataset: 55 YouTube matches, 4,830 annotated frames with 41-frame windows, player boxes, "
            "9 individual actions, 8 group activities. Large; needs a space with plenty of disk. "
            "If Google Drive refuses (quota), use --url with the old SFU zip link from the repo README.",
            folder="1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS"),
    Dataset("volleyball_ibrahim_ball", "gdrive_file",
            "Not stated; cite Perez et al., Pattern Recognition 2022 (skeleton-based relational reasoning)",
            "Ball locations for the Volleyball dataset frames (manual annotation).",
            files=[("1urZpZiiepC85JD1u3VeURgUpztRgI0yl", "volleyball_ball_annotations.zip")]),
    Dataset("volleyball_ibrahim_tracks", "gdrive_file",
            "Not stated; cite Sendo and Ukita, MVA 2019",
            "Manual player trajectories for all Volleyball dataset sequences.",
            files=[("1M-fXmAVw8WyFr30xb-LMi_Z-Qiv2nFzl", "volleyball_tracks_ukita.zip")]),
    Dataset("sportsmot", "manual_url", "CC BY-NC 4.0 (non-commercial)",
            "240 clips (basketball, volleyball, football), 720p 25 fps, player boxes and IDs. Sign up on Codalab, "
            "join the SportsMOT competition, copy the link under Participate > Get Data, pass it with --url.",
            commercial_ok=False),
    Dataset("soccernet_jersey", "soccernet", "SoccerNet terms (check soccer-net.org before commercial use)",
            "SoccerNet jersey-number dataset (task jersey-2023): player tracklets with the number per tracklet. "
            "The public training data for PARSeq-style jersey reading.",
            extra={"task": "jersey-2023", "splits": ["train", "test", "challenge"]}),
    Dataset("parseq_jersey_weights", "gdrive_file",
            "CC BY-NC (license.txt in mkoshkina/jersey-number-pipeline): non-commercial",
            "PARSeq weights from Koshkina and Elder (CVPRW 2024): original, hockey fine-tuned and SoccerNet "
            "fine-tuned. Goes to models/ rather than datasets/ if you pass --dest .../models.",
            files=[("1AK_GnM6pIYyfIf3tBYSKIyR3Fa3Z46Cx", "parseq_original.ckpt"),
                   ("1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE", "parseq_hockey.ckpt"),
                   ("1uRln22tlhneVt3P6MePmVxBWSLMsL3bm", "parseq_soccernet.ckpt")],
            commercial_ok=False),
    Dataset("jersey_str_lmdb", "gdrive_file",
            "CC BY-NC (mkoshkina/jersey-number-pipeline): non-commercial",
            "Weakly labelled jersey-number crops in LMDB format, used to fine-tune the text recogniser.",
            files=[("1PX8XDF3nNMZAvcjL6M5hurwX78ePAhSs", "jersey_str_lmdb.zip")],
            commercial_ok=False),
    Dataset("roboflow_ball_daniel", "roboflow", "See the dataset page on Roboflow Universe",
            "Volleyball ball detection (Roboflow Universe, daniel-eqnnu).",
            extra={"workspace": "daniel-eqnnu", "project": "volleyball_ball_object_detection_dataset-3jez2"}),
    Dataset("roboflow_ball_tracker", "roboflow", "See the dataset page on Roboflow Universe",
            "Volleyball ball tracker images (Roboflow Universe, test-rnrvz).",
            extra={"workspace": "test-rnrvz", "project": "volleyball-ball-tracker"}),
    Dataset("roboflow_volleyball01", "roboflow", "See the dataset page on Roboflow Universe",
            "Volleyball01 (Roboflow Universe, bumpsetcut).",
            extra={"workspace": "bumpsetcut", "project": "volleyball01"}),
    Dataset("roboflow_court", "roboflow", "See the dataset page on Roboflow Universe",
            "Volleyball court keypoints/detection (Roboflow Universe, protom) - a start for automatic court seeding.",
            extra={"workspace": "protom", "project": "volleyball-court-detection-tpvsi"}),
]}


# ------------------------------------------------------------------ destinations
class Dest:
    """s3://bucket/prefix or a local directory, with the few operations this script needs."""

    def __init__(self, uri: str):
        self.uri = uri.rstrip("/")
        self.is_s3 = uri.startswith("s3://")
        if self.is_s3:
            import boto3
            from boto3.s3.transfer import TransferConfig

            self.bucket, _, self.prefix = self.uri[5:].partition("/")
            self.s3 = boto3.client("s3")
            self.cfg = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=8)
        else:
            self.root = Path(uri)

    def _key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def size(self, rel: str) -> int | None:
        if self.is_s3:
            try:
                return self.s3.head_object(Bucket=self.bucket, Key=self._key(rel))["ContentLength"]
            except Exception:
                return None
        p = self.root / rel
        return p.stat().st_size if p.exists() else None

    def put_stream(self, rel: str, fileobj) -> None:
        if self.is_s3:
            self.s3.upload_fileobj(fileobj, self.bucket, self._key(rel), Config=self.cfg)
        else:
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                shutil.copyfileobj(fileobj, f, 8 << 20)

    def put_file(self, rel: str, path: Path) -> None:
        with open(path, "rb") as f:
            self.put_stream(rel, f)

    def put_json(self, rel: str, obj) -> None:
        import io

        self.put_stream(rel, io.BytesIO(json.dumps(obj, indent=1).encode()))

    def where(self, rel: str = "") -> str:
        return f"{self.uri}/{rel}" if rel else self.uri


class _Hashing:
    """File-like wrapper that hashes and counts bytes as they stream through."""

    def __init__(self, raw):
        self.raw, self.h, self.n = raw, hashlib.sha256(), 0

    def read(self, n=-1):
        b = self.raw.read(n)
        self.h.update(b)
        self.n += len(b)
        return b


def _file_record(rel: str, size: int, sha: str, source: str) -> dict:
    return {"key": rel, "bytes": size, "sha256": sha, "source": source}


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ fetchers
def fetch_http(d: Dataset, dest: Dest, force: bool, urls=None) -> list[dict]:
    out = []
    for url, rel in (urls or d.files):
        key = f"{d.name}/raw/{rel}"
        req = urllib.request.Request(url, headers={"User-Agent": "vb-datasets/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            if "drive" in url and "text/html" in (r.headers.get("Content-Type") or ""):
                raise RuntimeError(f"{rel}: Google Drive returned a web page, not the file "
                                   "(quota exceeded, access restricted or file moved); retry later or use gdown")
            length = int(r.headers.get("Content-Length") or -1)
            if not force and length > 0 and dest.size(key) == length:
                print(f"  skip {rel} (already there, {length:,} bytes)")
                out.append(_file_record(rel, length, "", url))
                continue
            hr = _Hashing(r)
            dest.put_stream(key, hr)
        print(f"  {rel}: {hr.n:,} bytes -> {dest.where(key)}")
        out.append(_file_record(rel, hr.n, hr.h.hexdigest(), url))
    return out


def _upload_tree(d: Dataset, dest: Dest, root: Path, force: bool, source: str, sub: str = "") -> list[dict]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = (Path(sub) / p.relative_to(root)).as_posix()
        key = f"{d.name}/raw/{rel}"
        size = p.stat().st_size
        if not force and dest.size(key) == size:
            out.append(_file_record(rel, size, "", source))
            continue
        dest.put_file(key, p)
        out.append(_file_record(rel, size, _sha_file(p), source))
    print(f"  {len(out)} files from {source} -> {dest.where(d.name + '/raw/' + sub)}")
    return out


def _extract_all(root: Path) -> None:
    for z in list(root.rglob("*.zip")):
        target = z.with_suffix("")
        print(f"  extracting {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(target)
        z.unlink()


def _have_gdown() -> bool:
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_gdrive_direct(d: Dataset, dest: Dest, force: bool) -> list[dict]:
    """Single Drive files streamed straight into the destination: no gdown, no scratch disk.
    Archives are stored as downloaded (no --extract); unzip them where you use them."""
    return fetch_http(d, dest, force, [(DRIVE_DIRECT.format(id=fid), rel) for fid, rel in d.files])


def fetch_gdrive(d: Dataset, dest: Dest, force: bool, scratch: Path, extract: bool) -> list[dict]:
    import gdown

    tmp = Path(tempfile.mkdtemp(prefix=d.name + "-", dir=scratch))
    try:
        if d.kind == "gdrive_folder":
            gdown.download_folder(id=d.folder, output=str(tmp), quiet=False, remaining_ok=True)
            src = f"https://drive.google.com/drive/folders/{d.folder}"
        else:
            for fid, rel in d.files:
                gdown.download(id=fid, output=str(tmp / rel), quiet=False)
            src = "https://drive.google.com/file/d/" + ",".join(f for f, _ in d.files)
        if extract:
            _extract_all(tmp)
        return _upload_tree(d, dest, tmp, force, src)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch_soccernet(d: Dataset, dest: Dest, force: bool, scratch: Path, extract: bool) -> list[dict]:
    from SoccerNet.Downloader import SoccerNetDownloader

    tmp = Path(tempfile.mkdtemp(prefix=d.name + "-", dir=scratch))
    try:
        SoccerNetDownloader(LocalDirectory=str(tmp)).downloadDataTask(task=d.extra["task"], split=d.extra["splits"])
        if extract:
            _extract_all(tmp)
        return _upload_tree(d, dest, tmp, force, f"SoccerNet task {d.extra['task']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch_roboflow(d: Dataset, dest: Dest, force: bool, scratch: Path, fmt: str) -> list[dict]:
    import roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("set ROBOFLOW_API_KEY (free account: app.roboflow.com > Settings > API Keys)")
    project = roboflow.Roboflow(api_key=key).workspace(d.extra["workspace"]).project(d.extra["project"])
    versions = project.versions()
    if not versions:
        raise RuntimeError(f"{d.name}: no published versions")
    ver = max(versions, key=lambda v: int(str(v.version).split("/")[-1]))
    vnum = int(str(ver.version).split("/")[-1])
    tmp = Path(tempfile.mkdtemp(prefix=d.name + "-", dir=scratch))
    try:
        ds = project.version(vnum).download(model_format=fmt, location=str(tmp / f"v{vnum}-{fmt}"))
        src = f"https://universe.roboflow.com/{d.extra['workspace']}/{d.extra['project']}/dataset/{vnum}"
        return _upload_tree(d, dest, Path(ds.location).parent, force, src)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ main
def run(names: list[str], dest_uri: str, force=False, extract=False, url=None, scratch=None, fmt="coco",
        direct=False) -> dict:
    dest = Dest(dest_uri)
    scratch = Path(scratch or tempfile.gettempdir())
    report = {}
    for name in names:
        d = REGISTRY[name]
        print(f"\n== {name} ({d.kind}) -> {dest.where(name)}")
        if d.commercial_ok is False:
            print(f"  NOTE licence: {d.licence}")
        if d.kind == "http":
            files = fetch_http(d, dest, force)
        elif d.kind in ("gdrive_folder", "gdrive_file"):
            if url:          # fallback mirror (e.g. the SFU zip for volleyball_ibrahim)
                files = fetch_http(d, dest, force, [(url, Path(urllib.parse.urlparse(url).path).name)])
            elif d.kind == "gdrive_file" and (direct or not _have_gdown()):
                files = fetch_gdrive_direct(d, dest, force)
            else:
                files = fetch_gdrive(d, dest, force, scratch, extract)
        elif d.kind == "manual_url":
            if not url:
                raise SystemExit(f"{name}: pass --url (see --list for where to get it)")
            files = fetch_http(d, dest, force, [(url, Path(urllib.parse.urlparse(url).path).name or "data.zip")])
        elif d.kind == "soccernet":
            files = fetch_soccernet(d, dest, force, scratch, extract)
        elif d.kind == "roboflow":
            files = fetch_roboflow(d, dest, force, scratch, fmt)
        else:
            raise ValueError(d.kind)
        manifest = {"name": d.name, "licence": d.licence, "commercial_ok": d.commercial_ok, "note": d.note,
                    "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "files": files, "total_bytes": sum(f["bytes"] for f in files)}
        dest.put_json(f"{name}/MANIFEST.json", manifest)
        report[name] = {"files": len(files), "bytes": manifest["total_bytes"]}
        print(f"  done: {len(files)} files, {manifest['total_bytes'] / 1e6:,.1f} MB, manifest at {dest.where(name + '/MANIFEST.json')}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="datasets to fetch (see --list); 'open' = all that need no sign-up")
    ap.add_argument("--dest", help="s3://bucket/datasets or a local directory")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-upload files already present")
    ap.add_argument("--extract", action="store_true", help="unzip archives before uploading")
    ap.add_argument("--url", help="direct link for manual_url datasets, or a mirror for a Drive dataset")
    ap.add_argument("--scratch", help="scratch dir for gdown/SoccerNet/Roboflow (default: system temp)")
    ap.add_argument("--format", default="coco", help="Roboflow export format (coco, yolov8, ...)")
    ap.add_argument("--direct", action="store_true",
                    help="stream single Google Drive files straight to the destination (no gdown, no scratch disk)")
    a = ap.parse_args(argv)
    if a.list or not a.names:
        for d in REGISTRY.values():
            flag = {True: "commercial use OK", False: "NON-COMMERCIAL", None: "licence not stated"}[d.commercial_ok]
            print(f"{d.name:26s} {d.kind:13s} [{flag}] {d.licence}\n{'':26s} {d.note}\n")
        return 0
    if not a.dest:
        ap.error("--dest is required")
    names = a.names
    if names == ["open"]:
        names = [n for n, d in REGISTRY.items() if d.kind in ("http", "gdrive_folder", "gdrive_file")]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        ap.error(f"unknown dataset(s): {unknown}; see --list")
    print(json.dumps(run(names, a.dest, a.force, a.extract, a.url, a.scratch, a.format, a.direct), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

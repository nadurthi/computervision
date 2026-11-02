# Step 06 - read-only check of what is in s3://vb-data-493280347481/datasets and /models.
#
# Lists every object under datasets/ and models/, reads each dataset's MANIFEST.json, and reports per
# dataset: files in the manifest vs objects in S3, bytes, any manifest entry whose S3 object is missing
# or a different size, and for .zip/.ckpt files whether the first bytes are a zip header ("PK"), which
# catches a Google Drive error page saved under a file name. Changes nothing.
# CLI equivalent, for reference:
#   aws s3 ls s3://vb-data-493280347481/datasets/ --recursive --summarize --human-readable
#   aws s3 ls s3://vb-data-493280347481/models/ --recursive --summarize --human-readable
#   aws s3 cp s3://vb-data-493280347481/datasets/<name>/MANIFEST.json -
B, REGION = "vb-data-493280347481", "us-east-1"
objs = {}
for pref in ("datasets/", "models/"):
    lst = await call_boto3(service_name="s3", operation_name="ListObjectsV2", region_name=REGION,
                           params={"Bucket": B, "Prefix": pref})
    objs.update({o["Key"]: o["Size"] for o in lst.get("Contents", [])})
names = sorted({"/".join(k.split("/")[:2]) for k in objs if k.count("/") >= 2})
report = {}
for name in names:
    mkey = f"{name}/MANIFEST.json"
    if mkey not in objs:
        report[name] = {"manifest": "missing", "objects": sum(1 for k in objs if k.startswith(f"{name}/"))}
        continue
    m = await call_boto3(service_name="s3", operation_name="GetObject", region_name=REGION, params={"Bucket": B, "Key": mkey})
    body = m["Body"]
    body = body.read() if hasattr(body, "read") else body
    man = json.loads(body.decode() if isinstance(body, (bytes, bytearray)) else body)
    problems = []
    for f in man["files"]:
        k = f"{name}/raw/{f['key']}"
        if k not in objs:
            problems.append(f"missing {k}")
        elif objs[k] != f["bytes"]:
            problems.append(f"size {k}: s3 {objs[k]} vs manifest {f['bytes']}")
        elif k.endswith((".zip", ".ckpt")):
            g = await call_boto3(service_name="s3", operation_name="GetObject", region_name=REGION,
                                 params={"Bucket": B, "Key": k, "Range": "bytes=0-3"})
            head = g["Body"]
            head = head.read() if hasattr(head, "read") else head
            head = head.encode("latin-1") if isinstance(head, str) else head
            if not head.startswith(b"PK"):
                problems.append(f"not a zip/checkpoint (first bytes {head!r}): {k}")
    raw = [k for k in objs if k.startswith(f"{name}/raw/")]
    report[name] = {"manifest_files": len(man["files"]), "s3_objects": len(raw), "bytes": man["total_bytes"],
                    "licence": man["licence"], "commercial_ok": man["commercial_ok"], "downloaded_at": man["downloaded_at"],
                    "problems": problems, "sha256": {f["key"]: f["sha256"] for f in man["files"]}}
for r in report.values():
    r.pop("sha256", None)                     # full checksums stay in the manifests; keep this summary short
result = {"objects": len(objs), "total_bytes": sum(objs.values()), "datasets": report}
result

#!/usr/bin/env bash
# S3 cache for the things every fresh instance would otherwise rebuild (S3 is the only persistent storage):
#   s3://<bucket>/cache/isaac-image-<tag>.tar.zst   the Isaac Sim Docker image (docker save, ~20 GB raw)
#   s3://<bucket>/cache/data-cache.tar.zst          run_all.sh's DATA folder: SimReady models + Isaac's shader,
#                                                   material and Kit caches (the slow first-start compile)
#
#   bash s3_cache.sh restore s3://bucket/cache /opt/isaac/data-cache nvcr.io/nvidia/isaac-sim:5.1.0
#   bash s3_cache.sh save    s3://bucket/cache /opt/isaac/data-cache nvcr.io/nvidia/isaac-sim:5.1.0
#
# restore: loads whatever is cached (missing entries are skipped; run_all.sh then downloads as usual).
# save:    uploads the image once per tag; uploads the data cache when it is missing or has grown by more than
#          5 % (new assets, new shaders) since the last upload. CACHE_REFRESH=1 forces both.
# Streams through zstd (multi-threaded) when available, so nothing extra is written to disk.
set -uo pipefail
CMD="${1:?restore|save}"; S3="${2:?s3://bucket/prefix}"; DATA="${3:?data folder}"; IMAGE="${4:?docker image}"
TAG=$(echo "$IMAGE" | sed 's#.*/##; s#[:/]#-#g')                  # isaac-sim-5.1.0
IMG_KEY="$S3/isaac-image-$TAG.tar.zst"; DATA_KEY="$S3/data-cache.tar.zst"; SIZE_KEY="$S3/data-cache.size"
log() { echo "[s3-cache] $*"; }
command -v zstd >/dev/null || apt-get install -y -qq zstd >/dev/null 2>&1 || true
if command -v zstd >/dev/null; then ZC="zstd -T0 -3 -q"; ZD="zstd -d -T0 -q"; else ZC="cat"; ZD="cat"; fi
exists() { aws s3 ls "$1" >/dev/null 2>&1; }

case "$CMD" in
restore)
  t0=$(date +%s)
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    log "image already present"
  elif exists "$IMG_KEY"; then
    log "loading $IMAGE from $IMG_KEY"
    aws s3 cp "$IMG_KEY" - | $ZD | docker load | tail -1 || log "image restore failed (run_all.sh will pull it)"
  else
    log "no cached image yet"
  fi
  t1=$(date +%s)
  if exists "$DATA_KEY"; then
    log "restoring $DATA from $DATA_KEY"
    mkdir -p "$DATA"
    aws s3 cp "$DATA_KEY" - | $ZD | tar -x -C "$DATA" || log "data restore failed (run_all.sh will download)"
  else
    log "no cached data yet"
  fi
  log "restore done: image $((t1 - t0)) s, data $(( $(date +%s) - t1 )) s"
  ;;
save)
  t0=$(date +%s)
  if [ "${CACHE_REFRESH:-0}" = "1" ] || ! exists "$IMG_KEY"; then
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
      log "uploading $IMAGE to $IMG_KEY"
      docker save "$IMAGE" | $ZC | aws s3 cp - "$IMG_KEY" --expected-size 40000000000 --only-show-errors \
        || log "image upload failed"
    fi
  else
    log "image already cached"
  fi
  now=$(du -sb "$DATA" 2>/dev/null | cut -f1); now=${now:-0}
  old=$(aws s3 cp "$SIZE_KEY" - 2>/dev/null || echo 0)
  if [ "${CACHE_REFRESH:-0}" = "1" ] || ! exists "$DATA_KEY" || [ "$now" -gt $(( old + old / 20 )) ]; then
    log "uploading $DATA ($((now / 1000000)) MB, was $((old / 1000000)) MB) to $DATA_KEY"
    tar -C "$DATA" -cf - . | $ZC | aws s3 cp - "$DATA_KEY" --expected-size 20000000000 --only-show-errors \
      && echo "$now" | aws s3 cp - "$SIZE_KEY" --only-show-errors || log "data upload failed"
  else
    log "data cache up to date ($((now / 1000000)) MB)"
  fi
  log "save done in $(( $(date +%s) - t0 )) s"
  ;;
*) echo "usage: $0 restore|save s3://bucket/prefix DATA IMAGE"; exit 1 ;;
esac

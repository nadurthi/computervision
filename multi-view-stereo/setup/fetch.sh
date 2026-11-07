#!/usr/bin/env bash
# Download helpers with an S3 mirror. Source this file. If CACHE_S3 is set (s3://bucket/prefix), every artefact is
# first looked up there and uploaded there after a successful download, so later sessions never depend on Google
# Drive quotas or slow academic servers again.
#
#   fetch_url <cache-name> <url> <dest-file>
#   fetch_gdrive <cache-name> <file-id> <dest-file>
#   fetch_gdrive_folder <cache-name> <folder-url> <dest-dir>

export PATH="$HOME/.local/bin:$PATH"
_gdown() {
  command -v gdown >/dev/null 2>&1 || uv tool install -q gdown >/dev/null 2>&1 || pip install -q --user gdown
  gdown "$@"
}

_cache_get() {   # _cache_get <name> <dest> -> 0 if restored from S3
  [ -n "${CACHE_S3:-}" ] || return 1
  aws s3 ls "$CACHE_S3/$1" >/dev/null 2>&1 || return 1
  aws s3 cp --only-show-errors "$CACHE_S3/$1" "$2" && echo "[fetch] $1 <- $CACHE_S3"
}

_cache_put() {   # _cache_put <name> <file>
  [ -n "${CACHE_S3:-}" ] || return 0
  aws s3 cp --only-show-errors "$2" "$CACHE_S3/$1" && echo "[fetch] $1 -> $CACHE_S3" || true
}

fetch_url() {
  local name=$1 url=$2 dest=$3
  mkdir -p "$(dirname "$dest")"
  [ -s "$dest" ] && return 0
  _cache_get "$name" "$dest" && return 0
  echo "[fetch] $name <- $url"
  if wget -q --tries=3 --timeout=60 -O "$dest.part" "$url" || curl -fsSL --retry 3 -o "$dest.part" "$url"; then
    mv "$dest.part" "$dest"
    _cache_put "$name" "$dest"
  else
    rm -f "$dest.part"
    echo "[fetch] FAILED $name ($url)" >&2
    return 1
  fi
}

fetch_gdrive() {
  local name=$1 id=$2 dest=$3
  mkdir -p "$(dirname "$dest")"
  [ -s "$dest" ] && return 0
  _cache_get "$name" "$dest" && return 0
  echo "[fetch] $name <- Google Drive $id"
  for attempt in 1 2 3; do
    if _gdown -q "$id" -O "$dest" && [ -s "$dest" ]; then
      _cache_put "$name" "$dest"
      return 0
    fi
    sleep $((attempt * 30))
  done
  echo "[fetch] FAILED $name: Google Drive refused the download (quota). Download it in a browser from" >&2
  echo "        https://drive.google.com/file/d/$id and upload it to ${CACHE_S3:-<CACHE_S3>}/$name" >&2
  return 1
}

fetch_gdrive_folder() {
  local name=$1 url=$2 dest=$3 tgz
  tgz=$(mktemp --suffix .tgz)
  mkdir -p "$dest"
  if _cache_get "$name.tgz" "$tgz"; then
    tar -xzf "$tgz" -C "$dest"; rm -f "$tgz"; return 0
  fi
  echo "[fetch] $name <- Google Drive folder $url"
  for attempt in 1 2 3; do
    if _gdown -q --folder "$url" -O "$dest" && [ -n "$(ls -A "$dest")" ]; then
      tar -czf "$tgz" -C "$dest" . && _cache_put "$name.tgz" "$tgz"
      rm -f "$tgz"
      return 0
    fi
    sleep $((attempt * 30))
  done
  rm -f "$tgz"
  echo "[fetch] FAILED $name: download the folder $url in a browser, tar it (tar -czf $name.tgz -C <folder> .)" >&2
  echo "        and upload it to ${CACHE_S3:-<CACHE_S3>}/$name.tgz" >&2
  return 1
}

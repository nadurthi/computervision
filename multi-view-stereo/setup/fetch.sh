#!/usr/bin/env bash
# Download helpers with an optional local cache. Source this file.
#
# If CACHE_DIR is set (any directory), every downloaded artefact is kept there under a stable name, and later
# downloads of the same artefact are served from it (hard link when possible, otherwise copy). Point CACHE_DIR at a
# disk that survives re-installs, or at a synced folder; the AWS runner syncs it with an S3 bucket.
#
#   fetch_url <cache-name> <url> <dest-file>
#   fetch_gdrive <cache-name> <file-id> <dest-file>
#   fetch_gdrive_folder <cache-name> <folder-url> <dest-dir>

export PATH="$HOME/.local/bin:$PATH"
_gdown() {
  command -v gdown >/dev/null 2>&1 || uv tool install -q gdown >/dev/null 2>&1 || pip install -q --user gdown
  gdown "$@"
}

_cache_get() {   # _cache_get <name> <dest> -> 0 if restored from CACHE_DIR
  [ -n "${CACHE_DIR:-}" ] && [ -s "$CACHE_DIR/$1" ] || return 1
  mkdir -p "$(dirname "$2")"
  ln -f "$CACHE_DIR/$1" "$2" 2>/dev/null || cp "$CACHE_DIR/$1" "$2"
  echo "[fetch] $1 <- cache"
}

_cache_put() {   # _cache_put <name> <file>
  [ -n "${CACHE_DIR:-}" ] || return 0
  mkdir -p "$CACHE_DIR"
  [ -s "$CACHE_DIR/$1" ] && return 0
  ln -f "$2" "$CACHE_DIR/$1" 2>/dev/null || cp "$2" "$CACHE_DIR/$1"
  echo "[fetch] $1 -> cache"
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
  echo "        https://drive.google.com/file/d/$id and save it as ${CACHE_DIR:-<CACHE_DIR>}/$name" >&2
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
  echo "        and save it as ${CACHE_DIR:-<CACHE_DIR>}/$name.tgz" >&2
  return 1
}

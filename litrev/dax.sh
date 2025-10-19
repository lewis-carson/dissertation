#!/bin/sh
# dax.sh - find arXiv links/IDs in a source file and download PDFs into a destination folder
# Usage: dax.sh [--dry-run] SOURCE_FILE DEST_DIR
# Example: ./dax.sh main.md litrev/

set -eu

usage() {
  cat <<EOF
Usage: $0 [--dry-run] SOURCE_FILE DEST_DIR

Find arXiv links or IDs inside SOURCE_FILE and download the corresponding PDFs
into DEST_DIR. If --dry-run is provided, the script will just print the actions
it would take.
EOF
  exit 1
}

if [ "${1-}" = "--help" ] || [ "${1-}" = "-h" ]; then
  usage
fi

dry_run=0
if [ "${1-}" = "--dry-run" ]; then
  dry_run=1
  shift
fi

if [ "$#" -ne 2 ]; then
  usage
fi

src="$1"
dest="$2"

if [ ! -f "$src" ]; then
  echo "Source file not found: $src" >&2
  exit 2
fi

mkdir -p "$dest"

# Collect candidate arXiv identifiers from the source file.
# 1) direct arXiv URLs: https://arxiv.org/abs/... or /pdf/...
# 2) modern arXiv IDs like 1234.56789 or 1234.56789v2
# 3) legacy IDs like hep-th/9901001

ids_tmp=$(mktemp)
trap 'rm -f "$ids_tmp"' EXIT

# Extract from arxiv.org links and normalize (strip host, .pdf suffix, query strings,
# and trailing markdown punctuation like ")" or "](http") )
grep -Eoi 'https?://[^/]*arxiv.org/(abs|pdf)/[^\s)<>]+' "$src" 2>/dev/null | \
  sed -E 's#https?://[^/]*arxiv.org/(abs|pdf)/##; s#\.pdf$##' | sed -E 's#\?.*$##; s#[\)\]\(,;:<>].*$##' >>"$ids_tmp" || true

# Extract modern numeric IDs
grep -Eo '[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?' "$src" 2>/dev/null >>"$ids_tmp" || true

# Extract legacy IDs like cs/9901001 or hep-th/9901001
grep -Eo '[a-zA-Z\-]+\/[0-9]{7}(v[0-9]+)?' "$src" 2>/dev/null >>"$ids_tmp" || true

# Normalize candidates into canonical arXiv IDs (modern numeric IDs or legacy IDs).
# For each line, extract the first match of either:
#  - modern: 1234.56789 or 1234.56789v2
#  - legacy: hep-th/9901001
ids=$(
  while IFS= read -r line; do
    # strip common trailing junk
    line=$(printf '%s' "$line" | sed -E 's/\?.*$//; s/[\)\]\(,;:<>].*$//; s/^[[:space:]]+|[[:space:]]+$//')
    # try modern ID
    id=$(printf '%s' "$line" | grep -Eo '[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?' || true)
    if [ -n "$id" ]; then
      printf '%s\n' "$id"
      continue
    fi
    # try legacy id
    id=$(printf '%s' "$line" | grep -Eo '[a-zA-Z\-]+\/[0-9]{7}(v[0-9]+)?' || true)
    if [ -n "$id" ]; then
      printf '%s\n' "$id"
      continue
    fi
    # nothing matched -> skip
  done < "$ids_tmp" | sort -u
)

if [ -z "$(printf "%s" "$ids")" ]; then
  echo "No arXiv links or IDs found in $src"
  exit 0
fi

echo "Found the following arXiv IDs:"
printf '%s
' "$ids"

echo

download_url() {
  id="$1"
  # Prefer the arxiv.org/pdf/ID.pdf URL.
  printf 'https://arxiv.org/pdf/%s.pdf' "$id"
}

failed=0
for id in $ids; do
  out="$dest/${id}.pdf"
  if [ -f "$out" ]; then
    echo "SKIP: $out already exists"
    continue
  fi

  url=$(download_url "$id")

  if [ "$dry_run" -eq 1 ]; then
    echo "DRY-RUN: would download $url -> $out"
    continue
  fi

  echo "Downloading $id -> $out"
  # Try curl first, fallback to wget if curl is unavailable.
  if command -v curl >/dev/null 2>&1; then
    if ! curl -fL --retry 3 -o "$out" "$url"; then
      echo "Primary download failed for $id, trying export.arxiv.org..." >&2
      if ! curl -fL --retry 3 -o "$out" "https://export.arxiv.org/pdf/${id}.pdf"; then
        echo "Failed to download $id" >&2
        rm -f "$out" || true
        failed=1
      fi
    fi
  elif command -v wget >/dev/null 2>&1; then
    if ! wget -q -O "$out" "$url"; then
      echo "Primary download failed for $id, trying export.arxiv.org..." >&2
      if ! wget -q -O "$out" "https://export.arxiv.org/pdf/${id}.pdf"; then
        echo "Failed to download $id" >&2
        rm -f "$out" || true
        failed=1
      fi
    fi
  else
    echo "Neither curl nor wget is installed. Cannot download files." >&2
    exit 3
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "Some downloads failed." >&2
  exit 4
fi

echo "Done. PDFs are in: $dest"

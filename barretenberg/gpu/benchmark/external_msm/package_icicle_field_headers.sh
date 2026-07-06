#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /path/to/open-icicle/icicle [/path/to/icicle-v4-field-only-headers.tar.gz]" >&2
  exit 2
fi

ICICLE_DIR="$(cd "$1" && pwd)"
OUT_TAR="${2:-/tmp/icicle-v4-field-only-headers.tar.gz}"
ROOT_NAME="icicle-v4-field-only-headers"
WORK_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

copy_header() {
  local src="$1"
  local dst="$2"
  install -D -m 0644 "$ICICLE_DIR/$src" "$WORK_DIR/$ROOT_NAME/include/$dst"
}

copy_header "include/icicle/errors.h" "icicle/errors.h"
copy_header "include/icicle/fields/field.h" "icicle/fields/field.h"
copy_header "include/icicle/fields/params_gen.h" "icicle/fields/params_gen.h"
copy_header "include/icicle/fields/snark_fields/bn254_base.h" "icicle/fields/snark_fields/bn254_base.h"
copy_header "include/icicle/math/host_math.h" "icicle/math/host_math.h"
copy_header "include/icicle/math/modular_arithmetic.h" "icicle/math/modular_arithmetic.h"
copy_header "include/icicle/math/storage.h" "icicle/math/storage.h"
copy_header "include/icicle/utils/log.h" "icicle/utils/log.h"
copy_header "include/icicle/utils/modifiers.h" "icicle/utils/modifiers.h"
copy_header "include/icicle/utils/rand_gen.h" "icicle/utils/rand_gen.h"
copy_header "backend/cuda/include/cuda_math.h" "cuda_math.h"
copy_header "backend/cuda/include/gpu-utils/sharedmem.h" "gpu-utils/sharedmem.h"
copy_header "backend/cuda/include/ptx.h" "ptx.h"

find "$WORK_DIR/$ROOT_NAME" -type f | sort | sed "s#^$WORK_DIR/$ROOT_NAME/##" > "$WORK_DIR/$ROOT_NAME/MANIFEST.txt"
tar -C "$WORK_DIR" -czf "$OUT_TAR" "$ROOT_NAME"
echo "$OUT_TAR"

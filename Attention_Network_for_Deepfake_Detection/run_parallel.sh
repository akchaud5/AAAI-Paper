#!/usr/bin/env bash
set -euo pipefail

OUT="./celebdb"
CONF="${CONF:-0.6}"
INTERVAL="${INTERVAL:-0.5}"
MIN_FR="${MIN_FR:-8}"
MAX_FR="${MAX_FR:-100}"

for D in Celeb-real Celeb-synthesis YouTube-real; do
  if [ -d "$D" ]; then
    echo ">> Launching: $D"
    python3 extract_faces_from_videos.py \
      -i "./$D" -o "$OUT" \
      --frames auto --interval_s "$INTERVAL" \
      --min_frames "$MIN_FR" --max_frames "$MAX_FR" \
      --max_per_dir none --conf "$CONF" &
  fi
done
wait
echo "All parallel jobs finished."

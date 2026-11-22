#!/usr/bin/env bash
# Usage: ./make_video.sh track.mp3 [output.mp4]
#   env: CHUNKS=2  (parallel render processes, one per CPU core)
#        CRF=19    (19 = master quality, 25 = smaller file for sharing)
# Analyses the track, renders the visual in parallel chunks, joins them
# losslessly and muxes the original audio. Works from any folder; temporary
# files go in a scratch folder that is deleted afterwards.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 track.mp3 [output.mp4]"; exit 1; }

abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
AUDIO="$(abspath "$1")"
OUT="$(abspath "${2:-visualiser.mp4}")"
HERE="$(cd "$(dirname "$0")" && pwd)"
CHUNKS="${CHUNKS:-2}"
CRF="${CRF:-19}"
[ -f "$AUDIO" ] || { echo "no such file: $AUDIO"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

python3 "$HERE/analyse.py" "$AUDIO" features.npz
DUR=$(python3 -c "import numpy as np; print(float(np.load('features.npz')['duration']))")

: > parts.txt
pids=()
for ((c = 0; c < CHUNKS; c++)); do
  S=$(python3 -c "print(round($DUR*$c/$CHUNKS, 3))")
  END=()
  if ((c < CHUNKS - 1)); then
    END=(--end "$(python3 -c "print(round($DUR*($c+1)/$CHUNKS, 3))")")
  fi
  python3 "$HERE/render.py" features.npz --start "$S" "${END[@]}" --crf "$CRF" \
          --out "part$c.mp4" > "part$c.log" 2>&1 &
  pids+=($!)
  echo "file 'part$c.mp4'" >> parts.txt
done
for ((c = 0; c < CHUNKS; c++)); do
  wait "${pids[$c]}" || { echo "chunk $c failed:"; tail -20 "part$c.log"; exit 1; }
done

ffmpeg -y -loglevel error -f concat -safe 0 -i parts.txt -c copy video_only.mp4
ffmpeg -y -loglevel error -i video_only.mp4 -i "$AUDIO" -map 0:v -map 1:a \
       -c:v copy -c:a aac -b:a 256k -shortest -movflags +faststart "$OUT.partial.mp4"
mv "$OUT.partial.mp4" "$OUT"
echo "wrote $OUT"
